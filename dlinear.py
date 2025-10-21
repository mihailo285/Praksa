import os
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from re import X
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from sklearn.metrics import r2_score




warnings.filterwarnings('ignore')

def time_features(dates, freq='15min'):
    """
    Creates time-based features from a datetime index.
    :param dates: array-like of datetime objects
    :param freq: 'h', 'd', 'min', etc.
    :return: np.array [feature, time_len]
    """
    dates = pd.to_datetime(dates)
    features = []

    if freq in ['h', 't', '15min', '30min', 'min']:
        features.append(dates.hour / 23.0)        # scaled hour
    if freq in ['d', 'h', 't', '15min', '30min', 'min']:
        features.append(dates.day / 31.0)         # scaled day of month
        features.append(dates.weekday / 6.0)    # scaled weekday
    if freq in ['m', 'd', 'h', 't', '15min', '30min', 'min']:
        features.append(dates.month / 12.0)       # scaled month

    return np.array(features)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='data_with_features_separated.csv',
                 target='FLOW', scale = True, timeenc=0, freq='min'):
        # size [seq_len, label_len, pred_len]
        # info

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        df_raw['TIMESTAMP'] = pd.to_datetime(df_raw['TIMESTAMP'])

        # 2. Postavljamo 'TIMESTAMP' kao indeks DataFrame-a.
        df_raw = df_raw.set_index('TIMESTAMP')

        # 3. Radimo resample. '15T' znači 15-minutna frekvencija.
        #    .mean() znači da će vrednosti unutar svakog intervala od 15 minuta biti usrednjene.
        #    Možete koristiti i .sum(), .first(), .last() itd. u zavisnosti od logike vaših podataka.

        df_raw = df_raw.resample('15T').mean()

        # 4. Uklanjamo redove koji su možda nastali sa NaN vrednostima ako je bilo rupa u podacima.
        df_raw.dropna(inplace=True)

        # 5. Vraćamo 'TIMESTAMP' iz indeksa nazad u regularnu kolonu.
        df_raw = df_raw.reset_index()

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('TIMESTAMP')
        df_raw = df_raw[['TIMESTAMP'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.timestamps = df_raw['TIMESTAMP'][border1:border2].values

        df_stamp = df_raw[['TIMESTAMP']][border1:border2].copy()
        df_stamp['TIMESTAMP'] = pd.to_datetime(df_stamp['TIMESTAMP'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['TIMESTAMP'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['TIMESTAMP'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['TIMESTAMP'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['TIMESTAMP'].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['TIMESTAMP'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['TIMESTAMP'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)



        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('TIMESTAMP')
        df_raw = df_raw[['TIMESTAMP'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['TIMESTAMP']][border1:border2]
        tmp_stamp['TIMESTAMP'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['TIMESTAMP'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['TIMESTAMP'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['TIMESTAMP'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['TIMESTAMP'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['TIMESTAMP'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['TIMESTAMP'].apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp['TIMESTAMP'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['TIMESTAMP'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['TIMESTAMP'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# MODEL

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            #self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                #self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            #self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]

# ==============================================================================
# === KORAK 1: KONFIGURACIJA I HIPERPARAMETRI ===
# ==============================================================================
class Configs:
    def __init__(self):
        # Parametri podataka i modela (MORAJU odgovarati podacima!)
        self.timeenc = 1
        self.root_path = '.'
        self.data_path = 'data_with_features_separated.csv'
        self.target = 'FLOW'
        self.features = 'MS'  # M, S, ili MS
        self.freq = '15min'  # Frekvencija nakon downsamplinga

        # Koristimo NOVE, preračunate vrednosti nakon downsamplinga
        self.seq_len = 1344
        self.label_len = 672
        self.pred_len = 4  # Predikcija za 2*15=30min

        # Broj karakteristika (features). Morate ga prilagoditi vašem CSV fajlu.
        # Npr. ako imate 'TIMESTAMP', 'FLOW' i još 5 karakteristika, onda je enc_in = 6 (FLOW + 5 ostalih)
        self.enc_in = 6
        self.individual = False  # Da li da se koristi poseban linearni sloj za svaku karakteristiku

        # Parametri treninga
        self.batch_size = 64  # Prilagodite na osnovu VRAM-a (32, 64, 128...)
        self.learning_rate = 0.00001
        self.num_epochs = 10  # Počnite sa 10, pa povećajte
        self.use_gpu = True


configs = Configs()

# ==============================================================================
# === KORAK 2: PRIPREMA PODATAKA ===
# ==============================================================================
print("Priprema podataka...")

# Kreiranje Dataset-a i DataLoader-a za svaki set
train_dataset = Dataset_Custom(
    root_path=configs.root_path, data_path=configs.data_path, flag='train',
    size=[configs.seq_len, configs.label_len, configs.pred_len],
    features=configs.features, target=configs.target, freq=configs.freq, scale=True, timeenc = configs.timeenc
)
val_dataset = Dataset_Custom(
    root_path=configs.root_path, data_path=configs.data_path, flag='val',
    size=[configs.seq_len, configs.label_len, configs.pred_len],
    features=configs.features, target=configs.target, freq=configs.freq, scale=True, timeenc = configs.timeenc
)
test_dataset = Dataset_Custom(
    root_path=configs.root_path, data_path=configs.data_path, flag='test',
    size=[configs.seq_len, configs.label_len, configs.pred_len],
    features=configs.features, target=configs.target, freq=configs.freq, scale=True, timeenc = configs.timeenc
)

train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=0, drop_last=False)

print("Podaci su spremni.")

# ==============================================================================
# === KORAK 3: INICIJALIZACIJA MODELA, OPTIMIZATORA I GUBITKA ===
# ==============================================================================
# Određivanje uređaja (GPU ili CPU)
device = torch.device("cuda:0" if configs.use_gpu and torch.cuda.is_available() else "cpu")
print(f"Koristi se uređaj: {device}")

# Inicijalizacija modela
model = Model(configs).to(device)

# Funkcija gubitka (Loss function) - Mean Squared Error je standard za regresiju
criterion = nn.MSELoss()

# Optimizator - Adam je robustan i dobar izbor za početak
optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

# ==============================================================================
# === KORAK 4: TRENIRANJE MODELA ===
# ==============================================================================
print("Početak treninga...")
train_start_time = time.time()

for epoch in range(configs.num_epochs):
    # --- PODGRUPA ZA TRENIRANJE MODELA ---
    epoch_start_time = time.time()
    model.train()  # Prebacivanje modela u mod za treniranje
    total_train_loss = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        optimizer.zero_grad()  # Nuliramo gradijente pre novog prolaska

        # Prebacivanje podataka na odabrani uređaj (GPU/CPU)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        # batch_x_mark i batch_y_mark se ne koriste u ovoj implementaciji DLinear modela

        # Forward pass: Prosleđujemo ulazne podatke modelu
        outputs = model(batch_x)

        # Ključni korak: Isecamo stvarni `y` da odgovara dužini predikcije
        # Model predviđa samo poslednjih `pred_len` koraka
        f_dim = -1  # -1 za multivarijantno, 0 za univarijantno
        true_y = batch_y[:, -configs.pred_len:, f_dim:].to(device)

        # Računanje gubitka
        loss = criterion(outputs, true_y)
        total_train_loss += loss.item()

        # Backward pass i optimizacija
        loss.backward()
        optimizer.step()

    # --- PODGRUPA ZA VALIDACIJU MODELA ---
    # --- Validacija nakon svake epohe ---
    model.eval()  # Prebacivanje modela u mod za evaluaciju
    total_val_loss = 0
    with torch.no_grad():  # Isključujemo računanje gradijenata za bržu evaluaciju
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)

            f_dim = -1
            true_y = batch_y[:, -configs.pred_len:, f_dim:].to(device)

            loss = criterion(outputs, true_y)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    epoch_time = time.time() - epoch_start_time

    print(
        f"Epoha {epoch + 1}/{configs.num_epochs} | Vreme: {epoch_time:.2f}s | Gubitak (Trening): {avg_train_loss:.7f} | Gubitak (Validacija): {avg_val_loss:.7f}")

print(f"\nTrening završen. Ukupno vreme: {(time.time() - train_start_time) / 60:.2f} min")

# ==============================================================================
# === KORAK 5: EVALUACIJA NA TEST SETU ===
# ==============================================================================
print("\nPočetak evaluacije na test setu...")
model.eval()

preds = []
trues = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        outputs = model(batch_x)

        true_y = batch_y[:, -configs.pred_len:, :]

        preds.append(outputs.detach().cpu().numpy())
        trues.append(true_y.detach().cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

# Inverzna transformacija na podacima sa SVIM karakteristikama
# Prvo ih preoblikujemo u 2D niz, kako skaler očekuje
preds_reshaped = preds.reshape(-1, preds.shape[-1])
trues_reshaped = trues.reshape(-1, trues.shape[-1])

# Sada radimo inverznu transformaciju
preds_inv = test_dataset.inverse_transform(preds_reshaped)
trues_inv = test_dataset.inverse_transform(trues_reshaped)

# Vraćamo ih u originalni 3D oblik (batch, pred_len, features)
preds_inv = preds_inv.reshape(preds.shape)
trues_inv = trues_inv.reshape(trues.shape)

# Tek SADA biramo samo ciljnu kolonu za računanje greške
target_col_idx = -1 # Pretpostavka da je ciljna promenljiva poslednja kolona
mse = np.mean((preds_inv[:, :, target_col_idx] - trues_inv[:, :, target_col_idx]) ** 2)
mae = np.mean(np.abs(preds_inv[:, :, target_col_idx] - trues_inv[:, :, target_col_idx]))

avg_flow = np.mean(trues_inv[:, :, -1])
std_flow = np.std(trues_inv[:, :, -1])

# RACUNANJE R2 SCORE

y_true_target = trues_inv[:, :, target_col_idx]
y_pred_target = preds_inv[:, :, target_col_idx]

y_true_flat = y_true_target.flatten()
y_pred_flat = y_pred_target.flatten()
r2 = r2_score(y_true_flat, y_pred_flat)

print(f"\nProsečna stvarna vrednost protoka (FLOW) na test setu: {avg_flow:.2f}")
print(f"Standardna devijacija protoka (FLOW) na test setu: {std_flow:.2f}")

print(f"Rezultati na test setu:")
print(f"MSE: {mse:.7f}")
print(f"MAE: {mae:.7f}")
print(f"R2 Score: {r2:.7f}")

# ==============================================================================
# === KORAK 6: VIZUELIZACIJA REZULTATA ===
# ==============================================================================
print("\nGenerisanje vizuelizacije predikcija za 'FLOW'...")

# Biramo jedan uzorak iz test seta za crtanje.
# Možete promeniti 'idx_to_plot' da vidite različite primere iz test seta.
idx_to_plot = 0

# --- Rekonstrukcija ulaznih podataka (Istorije) za odabrani uzorak ---
# Ovaj deo ostaje neophodan da bismo imali kontekst (sivu liniju) na grafikonu.
with torch.no_grad():
    batch_idx = idx_to_plot // configs.batch_size
    idx_in_batch = idx_to_plot % configs.batch_size

    for i, (batch_x, _, _, _) in enumerate(test_loader):
        if i == batch_idx:
            input_sequence_scaled = batch_x[idx_in_batch]
            input_sequence_inv = test_dataset.scaler.inverse_transform(input_sequence_scaled.numpy())
            break
# --- Kraj rekonstrukcije ---


# Izvlačimo predikcije i stvarne vrednosti za odabrani uzorak
# `trues_inv` i `preds_inv` su već izračunati i inverzno transformisani u KORAKU 5
y_true_sample_all_features = trues_inv[idx_to_plot]
y_pred_sample_all_features = preds_inv[idx_to_plot]


# Kreiramo JEDAN grafikon
plt.figure(figsize=(20, 6))

# Definišemo x-osu (vremenske korake)
input_steps = np.arange(configs.seq_len)
output_steps = np.arange(configs.seq_len, configs.seq_len + configs.pred_len)

# Definišemo indeks ciljne kolone ('FLOW'), što je -1 (poslednja)
target_col_idx = -1

# 1. Crtamo istoriju SAMO za 'FLOW'
plt.plot(input_steps, input_sequence_inv[:, target_col_idx], label='Ulazni podaci (Istorija za FLOW)', color='gray', alpha=0.7)

# 2. Crtamo stvarne buduće vrednosti SAMO za 'FLOW'
plt.plot(output_steps, y_true_sample_all_features[:, target_col_idx], label='Stvarna budućnost (FLOW)', color='blue', marker='.')

# 3. Crtamo ono što je model predvideo SAMO za 'FLOW'
plt.plot(output_steps, y_pred_sample_all_features[:, target_col_idx], label='Predviđena budućnost (FLOW)', color='red', linestyle='--', marker='x')

# Dodajemo naslov i oznake
plt.title(f'Predikcija vs. Stvarnost za ciljnu promenljivu: {configs.target}', fontsize=16)
plt.xlabel(f'Vremenski koraci (Interval od {configs.freq})')
plt.ylabel('Vrednost protoka (Originalna skala)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Prikazujemo finalni grafikon
plt.show()