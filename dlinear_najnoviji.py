import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import torch.nn as nn
import time
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
import random
import joblib

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


warnings.filterwarnings('ignore')

def time_features(dates, freq='min'):

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

class EarlyStopping:
    """Zaustavlja trening ako se validacioni gubitak ne poboljša nakon datog broja epoha (patience)."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping brojač: {self.counter} od {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Čuva model kada se validacioni gubitak smanji.'''
        if self.verbose:
            self.trace_func(f'Validacioni gubitak se smanjio ({self.val_loss_min:.6f} --> {val_loss:.6f}). Čuvanje modela u fajl: {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='data_with_features_separated.csv',
                 target='FLOW', scale = True, timeenc=0, freq='min', gap_len = 0, ):
        # size [seq_len, label_len, pred_len]
        # info
        print(f"\n>>> [Dataset] Inicijalizacija za '{flag}' set...")

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        print(f"    - Dužina ulaza (seq_len): {self.seq_len}, Dužina predikcije (pred_len): {self.pred_len}")
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.gap_len = gap_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        print(f">>> [Dataset] Početak čitanja i obrade podataka za '{['TRAIN', 'VAL', 'TEST'][self.set_type]}'...")
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), decimal = ',')
        print(f"    - Originalni fajl učitan. Oblik: {df_raw.shape}")
        df_raw['TIMESTAMP'] = pd.to_datetime(df_raw['TIMESTAMP'])
        '''
        # 2. Postavljamo 'TIMESTAMP' kao indeks DataFrame-a.
        df_raw = df_raw.set_index('TIMESTAMP')
        print(f"    - Downsampling podataka na frekvenciju: {self.freq}")
        # 3. Radimo resample. '15T' znači 15-minutna frekvencija.
        #    .mean() znači da će vrednosti unutar svakog intervala od 15 minuta biti usrednjene.
        #    Možete koristiti i .sum(), .first(), .last() itd. u zavisnosti od logike vaših podataka.

        df_raw = df_raw.resample('min').mean()

        # 4. Uklanjamo redove koji su možda nastali sa NaN vrednostima ako je bilo rupa u podacima.
        df_raw.dropna(inplace=True)

        # 5. Vraćamo 'TIMESTAMP' iz indeksa nazad u regularnu kolonu.
        df_raw = df_raw.reset_index()
        print(f"    - Oblik nakon downsamplinga: {df_raw.shape}")
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('TIMESTAMP')
        df_raw = df_raw[['TIMESTAMP'] + cols + [self.target]]

        # print(cols)
        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.3)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x_start_index = border1

        if self.features == 'M':
            print(f"    - Mod rada 'M': Ulaz su sve karakteristike osim '{self.target}', izlaz je samo '{self.target}'.")
            cols_data_x = list(df_raw.columns)
            cols_data_x.remove('TIMESTAMP')
            cols_data_x.remove(self.target)
            cols_data_x.sort()
            df_data_x = df_raw[cols_data_x]
            df_data_y = df_raw[[self.target]]
            self.feature_names = cols_data_x
        else:  # 'MS' ili 'S'
            cols_data = df_raw.columns[1:]
            df_data_x = df_raw[cols_data]
            df_data_y = df_raw[cols_data]

        if self.scale:
            # === KLJUČNA IZMENA: Dva odvojena skalera ===
            print("    - Skaliranje podataka aktivirano (StandardScaler).")
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()

            train_data_x = df_data_x.iloc[border1s[0]:border2s[0]]
            self.scaler_x.fit(train_data_x.values)
            data_x = self.scaler_x.transform(df_data_x.values)

            train_data_y = df_data_y.iloc[border1s[0]:border2s[0]]
            self.scaler_y.fit(train_data_y.values)
            data_y = self.scaler_y.transform(df_data_y.values)
        else:
            data_x = df_data_x.values
            data_y = df_data_y.values

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

        self.timestamps = df_raw['TIMESTAMP'][border1:border2].values
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp
        print(f"    - Finalni oblik podataka za ovaj set: data_x={self.data_x.shape}, data_y={self.data_y.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end + self.gap_len
        r_end = r_begin + self.pred_len # Izlaz je samo budućnost

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.pred_len, 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len - self.gap_len + 1

    def inverse_transform(self, data, scaler_type='y'):
        if scaler_type == 'y':
            return self.scaler_y.inverse_transform(data)
        elif scaler_type == 'x':
            return self.scaler_x.inverse_transform(data)
        else:
            raise ValueError("scaler_type mora biti 'x' ili 'y'")

# MODEL

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        print(f"    - [moving_avg] Kreiran sloj za pokretni prosek sa veličinom prozora (kernel_size): {self.kernel_size}")
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
        print(f">>> [Decomposition] Kreiran blok za dekompoziciju serije.")
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
        print(f"\n>>> [Model] Kreiranje DLinear modela...")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self._forward_info_printed = False

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.c_out = configs.c_out

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
            self.Linear_Seasonal = nn.Linear(self.seq_len * self.channels, self.pred_len * self.c_out)
            self.Linear_Trend = nn.Linear(self.seq_len * self.channels, self.pred_len * self.c_out)

            # Inicijalizacija težina (opciono, ali dobra praksa)
            self.Linear_Seasonal.weight = nn.Parameter((1 / (self.seq_len * self.channels)) * torch.ones(
                [self.pred_len * self.c_out, self.seq_len * self.channels]))
            self.Linear_Trend.weight = nn.Parameter((1 / (self.seq_len * self.channels)) * torch.ones(
                [self.pred_len * self.c_out, self.seq_len * self.channels]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if not self._forward_info_printed:
            self._forward_info_printed = True

        seasonal_init, trend_init = self.decompsition(x)
        batch_size = seasonal_init.size(0)
        seasonal_init_flat = seasonal_init.reshape(batch_size, -1)
        trend_init_flat = trend_init.reshape(batch_size, -1)

        # Prosleđujemo "ispeglane" podatke kroz linearne slojeve
        seasonal_output_flat = self.Linear_Seasonal(seasonal_init_flat)
        trend_output_flat = self.Linear_Trend(trend_init_flat)

        # Sabiramo izlaze
        output_flat = seasonal_output_flat + trend_output_flat

        # Vraćamo "ispeglan" izlaz u željeni 3D oblik
        # Oblik se menja sa [Batch, pred_len * c_out] na [Batch, pred_len, c_out]
        x_out = output_flat.reshape(batch_size, self.pred_len, self.c_out)

        return x_out

# ==============================================================================
# === KORAK 1: KONFIGURACIJA I HIPERPARAMETRI ===
# ==============================================================================
class Configs:
    def __init__(self):
        # Parametri podataka i modela (MORAJU odgovarati podacima!)
        self.timeenc = 1
        self.root_path = '.'
        self.data_path = 'podaci_testiranje_kendall.csv'
        self.target = 'FLOW'
        self.features = 'M'  # M, S, ili MS
        self.freq = 'min'  # Frekvencija nakon downsamplinga

        # Koristimo NOVE, preračunate vrednosti nakon downsamplinga
        self.seq_len = 60 # 2 * 60
        self.label_len = 0
        self.gap_len = 0
        self.pred_len = 1  # Predikcija za 2*15=30min

        # Broj karakteristika (features). Morate ga prilagoditi vašem CSV fajlu.
        # Npr. ako imate 'TIMESTAMP', 'FLOW' i još 5 karakteristika, onda je enc_in = 6 (FLOW + 5 ostalih)
        self.enc_in = 5
        self.c_out = 1
        self.individual = False  # Da li da se koristi poseban linearni sloj za svaku karakteristiku

        # Parametri treninga
        self.batch_size = 32  # Prilagodite na osnovu VRAM-a (32, 64, 128...)
        self.learning_rate = 0.0005
        self.num_epochs = 30  # Počnite sa 10, pa povećajte
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
    features=configs.features, target=configs.target, freq=configs.freq, scale=True, timeenc = configs.timeenc, gap_len=configs.gap_len
)
print(f"Pokretanje funkcije time_features za kreiranje vremenskih karakteristika sa frekvencijom '{configs.freq}' za trening skup podataka")
val_dataset = Dataset_Custom(
    root_path=configs.root_path, data_path=configs.data_path, flag='val',
    size=[configs.seq_len, configs.label_len, configs.pred_len],
    features=configs.features, target=configs.target, freq=configs.freq, scale=True, timeenc = configs.timeenc, gap_len=configs.gap_len
)
print(f"Pokretanje funkcije time_features za kreiranje vremenskih karakteristika sa frekvencijom '{configs.freq}' za validacioni skup podataka")
test_dataset = Dataset_Custom(
    root_path=configs.root_path, data_path=configs.data_path, flag='test',
    size=[configs.seq_len, configs.label_len, configs.pred_len],
    features=configs.features, target=configs.target, freq=configs.freq, scale=True, timeenc = configs.timeenc, gap_len=configs.gap_len
)
print(f"Pokretanje funkcije time_features za kreiranje vremenskih karakteristika sa frekvencijom '{configs.freq}' za test skup podataka")

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
print(f"    - Funkcija gubitka (Criterion): {type(criterion).__name__}")

# Optimizator - Adam je robustan i dobar izbor za početak
optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
print(f"    - Optimizator: {type(optimizer).__name__}")
print(f"    - Stopa učenja (Learning Rate): {configs.learning_rate}")

print("\n--- Model i komponente su spremni za trening. ---")

# ==============================================================================
# === KORAK 4: TRENIRANJE MODELA ===
# ==============================================================================
print("Početak treninga...")
train_start_time = time.time()
save_path = 'dlinear_best_model_kendall.pth'
early_stopping = EarlyStopping(patience=10, verbose=True, path=save_path)

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
        true_y = batch_y[:, -configs.pred_len:, :].to(device)

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
            true_y = batch_y[:, -configs.pred_len:, :].to(device)

            loss = criterion(outputs, true_y)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    epoch_time = time.time() - epoch_start_time

    print(
        f"Epoha {epoch + 1}/{configs.num_epochs} | Vreme: {epoch_time:.2f}s | Gubitak (Trening): {avg_train_loss:.7f} | Gubitak (Validacija): {avg_val_loss:.7f}")

    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping - trening prekinut.")
        break  # Izlazimo iz glavne for petlje

print(f"\nTrening završen. Ukupno vreme: {(time.time() - train_start_time) / 60:.2f} min")


# ==============================================================================
# === KORAK 5: EVALUACIJA NA TEST SETU ===
# ==============================================================================
#print("\nPočetak evaluacije na test setu...")
#model.eval()
best_model = Model(configs).to(device)

# Zatim, učitavamo sačuvane težine u tu instancu
best_model.load_state_dict(torch.load(save_path))

# Sada, za sve naredne korake (KORAK 5 i 6), koristimo 'best_model' umesto originalnog 'model'
# Primer za KORAK 5:
print("\nPočetak evaluacije na test setu sa NAJBOLJIM modelom...")
best_model.eval()

preds = []
trues = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        outputs = best_model(batch_x)

        true_y = batch_y[:, -configs.pred_len:, :]

        preds.append(outputs.detach().cpu().numpy())
        trues.append(true_y.detach().cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

# Inverzna transformacija na podacima sa SVIM karakteristikama
# Prvo ih preoblikujemo u 2D niz, kako skaler očekuje
preds_reshaped = preds.reshape(-1, configs.c_out)
trues_reshaped = trues.reshape(-1, configs.c_out)

# Sada radimo inverznu transformaciju
preds_inv = test_dataset.inverse_transform(preds_reshaped, scaler_type='y')
trues_inv = test_dataset.inverse_transform(trues_reshaped, scaler_type='y')

# Tek SADA biramo samo ciljnu kolonu za računanje greške
target_col_idx = 0 # Pretpostavka da je ciljna promenljiva poslednja kolona

mse = mean_squared_error(trues_inv, preds_inv)
mae = mean_absolute_error(trues_inv, preds_inv)
r2 = r2_score(trues_inv, preds_inv)

avg_flow = np.mean(trues_inv)
std_flow = np.std(trues_inv)



print(f"\nProsečna stvarna vrednost protoka (FLOW) na test setu: {avg_flow:.2f}")
print(f"Standardna devijacija protoka (FLOW) na test setu: {std_flow:.2f}")

print(f"Rezultati na test setu:")
print(f"MSE: {mse:.7f}")
print(f"MAE: {mae:.7f}")
print(f"R2 Score: {r2:.7f}")


SPECIFIC_DATE_TO_ANALYZE = '2025-01-28 00:00'

test_timestamps = test_dataset.timestamps
first_valid_timestamp = test_timestamps[0]
last_valid_timestamp = test_timestamps[-(configs.seq_len + configs.pred_len)]

print("\n--- Analiza na Testnom Skupu ---")
print(f"Testni skup podataka obuhvata period od: {pd.Timestamp(test_timestamps[0]).strftime('%Y-%m-%d %H:%M')} do {pd.Timestamp(test_timestamps[-1]).strftime('%Y-%m-%d %H:%M')}")
print(f"Opseg datuma za koje se može izvršiti kompletna analiza: od {pd.Timestamp(first_valid_timestamp).strftime('%Y-%m-%d %H:%M')} do {pd.Timestamp(last_valid_timestamp).strftime('%Y-%m-%d %H:%M')}")
date_to_analyze_str = ''

if SPECIFIC_DATE_TO_ANALYZE:
    # Korisnik je zadao specifičan datum
    try:
        # Proveravamo da li je zadati datum unutar validnog opsega testnog skupa
        chosen_date = pd.to_datetime(SPECIFIC_DATE_TO_ANALYZE)
        if first_valid_timestamp <= chosen_date <= last_valid_timestamp:
            date_to_analyze_str = SPECIFIC_DATE_TO_ANALYZE
            print(f"Koristi se specifično zadat datum za analizu: {date_to_analyze_str}")
        else:
            print(f"!!! UPOZORENJE: Zadat datum '{SPECIFIC_DATE_TO_ANALYZE}' je van validnog opsega testnog skupa.")
            print(f"    - Nastavljam sa nasumično izabranim datumom.")
            SPECIFIC_DATE_TO_ANALYZE = None # Vraćamo na None da bi se izvršio nasumičan izbor
    except Exception as e:
        print(f"!!! GREŠKA: Nije moguće parsirati zadati datum '{SPECIFIC_DATE_TO_ANALYZE}'. Greška: {e}")
        print(f"    - Nastavljam sa nasumično izabranim datumom.")
        SPECIFIC_DATE_TO_ANALYZE = None # Vraćamo na None

if not SPECIFIC_DATE_TO_ANALYZE:
    # Biramo nasumičan datum jer specifičan nije zadat ili nije bio validan
    random_index = random.randint(0, len(test_timestamps) - (configs.seq_len + configs.pred_len) - 1)
    random_date_to_analyze = test_timestamps[random_index]
    date_to_analyze_str = random_date_to_analyze.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Nasumično izabran datum za detaljnu analizu: {date_to_analyze_str}")

def analyze_prediction_at_date(start_date_str, dataset_for_scalers, model, configs, device):

    print(f"\n--- Analiza za specifičan datum: Početak istorije = {start_date_str} ---")

    # === KORAK 1: Učitavanje i obrada sirovih podataka (Logika iz 'prepare_data') ===
    try:
        print(">>> Učitavanje i obrada sirovih podataka za analizu...")
        df_full = pd.read_csv(os.path.join(configs.root_path, configs.data_path), decimal = ',')
        df_full['TIMESTAMP'] = pd.to_datetime(df_full['TIMESTAMP'])
        '''df_full = df_full.set_index('TIMESTAMP')
        df_full.sort_index(inplace=True)

        df_full = df_full.resample(configs.freq).mean()

        df_full.fillna(method='ffill', inplace=True)
        df_full.dropna(inplace=True)

        df_full = df_full.reset_index()'''
        print(f">>> Podaci uspešno obrađeni. Finalni oblik: {df_full.shape}")
    except Exception as e:
        print(f"!!! GREŠKA pri učitavanju i obradi podataka: {e}")
        return

    # === KORAK 2: Pronalaženje indeksa u obrađenim podacima ===
    try:
        target_start_date = pd.to_datetime(start_date_str)
        available_timestamps = df_full['TIMESTAMP']
        target_idx = available_timestamps.searchsorted(target_start_date, side='left')

        if target_idx >= len(available_timestamps) or available_timestamps.iloc[
            target_idx].date() != target_start_date.date():
            print(
                f"!!! UPOZORENJE: Tačan datum {start_date_str} nije pronađen. Najbliži je {available_timestamps.iloc[target_idx]}.")
            return
        context_after_pred = configs.pred_len * 5  # Koliko podataka želimo da vidimo nakon predikcije
        required_length = configs.seq_len + configs.gap_len + configs.pred_len + context_after_pred
        if target_idx + required_length >= len(df_full):
            print(f"!!! UPOZORENJE: Nema dovoljno podataka nakon {start_date_str} za kompletnu analizu sa kontekstom.")
            return

    except Exception as e:
        print(f"!!! GREŠKA pri pronalaženju datuma: {e}")
        return

    print(f">>> Pronađen odgovarajući početni indeks u celom setu: {target_idx}")

    # === KORAK 3: Isecanje ulaznih i ciljnih podataka ===
    start_x = target_idx
    end_x = target_idx + configs.seq_len
    input_df = df_full.iloc[start_x:end_x]

    start_y = end_x + configs.gap_len
    end_y = start_y + configs.pred_len
    true_future_df = df_full.iloc[start_y:end_y]

    input_features_df = input_df.drop(columns=['TIMESTAMP', configs.target])

    feature_order = dataset_for_scalers.feature_names
    input_features_df = input_features_df[feature_order]

    # === KORAK 4: Skaliranje, predikcija i inverzna transformacija ===
    # KORISTIMO SKALERE NAUČENE NA TRENING SETU!
    input_scaled = dataset_for_scalers.scaler_x.transform(input_features_df.values)

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_scaled).unsqueeze(0).float().to(device)
        prediction_scaled = model(input_tensor).squeeze(0).cpu().numpy()

    predicted_future_inv = dataset_for_scalers.scaler_y.inverse_transform(prediction_scaled)

    # === KORAK 5: Priprema za vizuelizaciju i ispis ===
    # Isecamo duži prozor za crtanje plave linije
    start_plot = target_idx
    end_plot = target_idx + required_length
    plot_df = df_full.iloc[start_plot:end_plot]

    timestamps_for_plotting = plot_df['TIMESTAMP']
    y_values_for_plotting = plot_df[configs.target].values

    # Definišemo vremensku osu samo za našu predikciju
    start_pred_idx_local = configs.seq_len + configs.gap_len
    end_pred_idx_local = start_pred_idx_local + configs.pred_len
    pred_timestamps = timestamps_for_plotting[start_pred_idx_local:end_pred_idx_local]

    # Izdvajamo stvarne vrednosti za period predikcije za tabelarni ispis
    true_future_y_values = y_values_for_plotting[start_pred_idx_local:end_pred_idx_local]

    # === Tabelarni ispis (ostaje isti) ===
    # ... (kod za tabelarni ispis) ...
    print("\n>>> Uporedni prikaz stvarnih i predviđenih vrednosti:")
    print("-" * 70)
    print(f"{'Tačka':>5} | {'Datum i Vreme':^25} | {'Stvarna Vrednost':>15} | {'Predviđena Vrednost':>20}")
    print("-" * 70)
    for i in range(len(predicted_future_inv)):
        print(
            f"{i + 1:>5} | {pred_timestamps.iloc[i].strftime('%Y-%m-%d %H:%M'):^25} | {true_future_y_values[i]:>15.2f} | {predicted_future_inv[i][0]:>20.2f}")
    print("-" * 70 + "\n")

    # === KORAK 6: Crtanje grafikona ===
    fig, ax = plt.subplots(figsize=(20, 8))

    # Crtamo STVARNE vrednosti za ceo duži period
    ax.plot(timestamps_for_plotting, y_values_for_plotting, label=f'Stvarne vrednosti ({configs.target})', color='blue')

    # Crtamo predikciju
    ax.plot(pred_timestamps, predicted_future_inv.flatten(), label='Predviđene vrednosti', color='red', linestyle='--', marker = 'o', markersize = 3)

    # Označavamo "gap"
    if configs.gap_len > 0:
        gap_start_idx_local = configs.seq_len
        gap_end_idx_local = configs.seq_len + configs.gap_len
        gap_start_ts = timestamps_for_plotting.iloc[gap_start_idx_local]
        gap_end_ts = timestamps_for_plotting.iloc[gap_end_idx_local]
        ax.axvspan(gap_start_ts, gap_end_ts, color='orange', alpha=0.2, label=f'Razmak od {configs.gap_len} koraka')
    plt.title(f'Analiza Predikcije sa Početkom Istorije na {start_date_str}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')  # Format za dane
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.tight_layout()
    plt.show()
if date_to_analyze_str:
    analyze_prediction_at_date(
    start_date_str=date_to_analyze_str, # Specifičan datum i vreme
    dataset_for_scalers=test_dataset,
    model=best_model,
    configs=configs, # Prosleđujemo novu, minutnu konfiguraciju
    device=device
    )
else:
    print("\nNije bilo moguće izabrati datum za detaljnu analizu. Preskačem...")

try:
    joblib.dump(train_dataset.scaler_x, 'scaler_x_dlinear.pkl')
    joblib.dump(train_dataset.scaler_y, 'scaler_y_dlinear.pkl')
except AttributeError:
    print("!!! GREŠKA: 'train_dataset' nema atribute 'scaler_x' ili 'scaler_y'.")

