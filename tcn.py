import os
import numpy as np
import pandas as pd
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

# Postavljanje seed-a za reproduktivnost
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Podešavanje za OMP grešku (ako je potrebno)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

warnings.filterwarnings('ignore')


# ==============================================================================
# === SEKCIJA 1: TCN KLASE (Nove klase dodate na vrh) ===
# ==============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# === SEKCIJA 2: NOVA "OMOT" KLASA - Model_TCN ===
# ==============================================================================
class Model_TCN(nn.Module):
    """
    TCN model upakovan da bude kompatibilan sa ostatkom skripte.
    Ovo je ključna adaptacija.
    """

    def __init__(self, configs):
        super(Model_TCN, self).__init__()
        print(f"\n>>> [Model] Kreiranje TCN modela...")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self._forward_info_printed = False

        # Inicijalizacija TCN mreže
        self.tcn = TemporalConvNet(
            num_inputs=configs.enc_in,
            num_channels=configs.tcn_channels,
            kernel_size=configs.tcn_kernel_size,
            dropout=configs.dropout
        )
        print(f"    - [TCN] Kreiran TemporalConvNet sa {len(configs.tcn_channels)} sloja/eva.")

        # Linearni sloj koji mapira izlaz TCN-a u željenu predikciju
        # Izlaz TCN-a ima `configs.tcn_channels[-1]` kanala
        last_channel_size = configs.tcn_channels[-1]
        self.decoder = nn.Linear(last_channel_size, self.pred_len * self.c_out)
        print(f"    - [Decoder] Kreiran linearni sloj: {last_channel_size} -> {self.pred_len * self.c_out}")
        #self.init_weights()

    #def init_weights(self):
        #self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x ulazi u obliku: [Batch, Dužina sekvence, Broj kanala] -> npr. [32, 1440, 5]
        if not self._forward_info_printed:
            print(f">>> [TCN Forward] Prvi prolaz kroz model...")
            print(f"    - Ulazni oblik (original): {x.shape}")
            self._forward_info_printed = True

        # nn.Conv1d (koji TCN koristi) očekuje oblik: [Batch, Broj kanala, Dužina sekvence]
        # Zato moramo da permutujemo dimenzije 1 i 2
        x_permuted = x.permute(0, 2, 1)
        if self._forward_info_printed and not hasattr(self, '_permute_info_printed'):
            print(f"    - Oblik nakon permutacije za TCN: {x_permuted.shape}")
            self._permute_info_printed = True

        # Propustimo podatke kroz TCN
        y_tcn = self.tcn(x_permuted)  # Izlaz je oblika [Batch, Zadnji_kanal, Dužina sekvence]

        # Uzimamo izlaz samo iz POSLEDNJEG vremenskog koraka TCN-a
        # Ovaj "sažeti" vektor nosi informaciju o celoj sekvenci
        y_pooled = y_tcn.mean(dim=2)  # Oblik: [Batch, Zadnji_kanal]
        if self._forward_info_printed and not hasattr(self, '_last_step_info_printed'):
            print(f"    - Oblik nakon TCN-a: {y_tcn.shape}")
            print(f"    - Uzimamo samo poslednji vremenski korak za dekoder. Oblik: {y_pooled.shape}")
            self._last_step_info_printed = True

        # Prosleđujemo kroz finalni linearni sloj za predikciju
        output = self.decoder(y_pooled)  # Oblik: [Batch, pred_len * c_out]

        # Vraćamo izlaz u željeni 3D oblik [Batch, pred_len, c_out]
        # da bi se poklopio sa onim što `criterion` i evaluacija očekuju
        x_out = output.view(-1, self.pred_len, self.c_out)
        if self._forward_info_printed and not hasattr(self, '_final_shape_info_printed'):
            print(f"    - Oblik nakon dekodera: {output.shape}")
            print(f"    - Finalni izlazni oblik modela: {x_out.shape}")
            self._final_shape_info_printed = True

        return x_out


# ==============================================================================
# === SEKCIJA 3: DATASET I POMOĆNE KLASE (Nepromenjeno!) ===
# Ovde se nalaze klase: time_features, EarlyStopping, Dataset_Custom, Dataset_Pred
# koje ste već imali. One ostaju potpuno iste.
# ==============================================================================

def time_features(dates, freq='15min'):
    dates = pd.to_datetime(dates)
    features = []
    if freq in ['h', 't', '15min', '30min', 'min']: features.append(dates.hour / 23.0)
    if freq in ['d', 'h', 't', '15min', '30min', 'min']:
        features.append(dates.day / 31.0)
        features.append(dates.weekday / 6.0)
    if freq in ['m', 'd', 'h', 't', '15min', '30min', 'min']: features.append(dates.month / 12.0)
    return np.array(features)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience, self.verbose, self.counter, self.best_score, self.early_stop, self.val_loss_min, self.delta, self.path, self.trace_func = patience, verbose, 0, None, False, np.inf, delta, path, trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping brojač: {self.counter} od {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save_checkpoint(val_loss, model); self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose: self.trace_func(
            f'Validacioni gubitak se smanjio ({self.val_loss_min:.6f} --> {val_loss:.6f}). Čuvanje modela: {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='data_with_features_separated.csv',
                 target='FLOW', scale=True, timeenc=0, freq='min', gap_len=0):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 24 * 4 * 4, 24 * 4, 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        assert flag in ['train', 'test', 'val'];
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features, self.target, self.scale, self.timeenc, self.freq, self.gap_len, self.root_path, self.data_path = features, target, scale, timeenc, freq, gap_len, root_path, data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), decimal=',')
        df_raw['TIMESTAMP'] = pd.to_datetime(df_raw['TIMESTAMP'])
        df_raw = df_raw.set_index('TIMESTAMP').resample("min").mean().dropna().reset_index()
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('TIMESTAMP')
        df_raw = df_raw[['TIMESTAMP'] + cols + [self.target]]
        num_train, num_test = int(len(df_raw) * 0.6), int(len(df_raw) * 0.3)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + (len(df_raw) - num_train - num_test), len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        self.data_x_start_index = border1
        if self.features == 'M':
            cols_data_x = [c for c in df_raw.columns if c not in ['TIMESTAMP', self.target]]
            df_data_x, df_data_y = df_raw[cols_data_x], df_raw[[self.target]]
            self.feature_names = cols_data_x
        else:
            df_data_x, df_data_y = df_raw[df_raw.columns[1:]], df_raw[df_raw.columns[1:]]
        if self.scale:
            self.scaler_x, self.scaler_y = StandardScaler(), StandardScaler()
            self.scaler_x.fit(df_data_x.iloc[border1s[0]:border2s[0]].values)
            data_x = self.scaler_x.transform(df_data_x.values)
            self.scaler_y.fit(df_data_y.iloc[border1s[0]:border2s[0]].values)
            data_y = self.scaler_y.transform(df_data_y.values)
        else:
            data_x, data_y = df_data_x.values, df_data_y.values
        self.data_x, self.data_y = data_x[border1:border2], data_y[border1:border2]
        df_stamp = df_raw[['TIMESTAMP']][border1:border2].copy()
        df_stamp['TIMESTAMP'] = pd.to_datetime(df_stamp['TIMESTAMP'])
        if self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['TIMESTAMP'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.timestamps = df_raw['TIMESTAMP'][border1:border2].values


    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        r_begin, r_end = s_end + self.gap_len, s_end + self.gap_len + self.pred_len
        seq_x, seq_y = self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]
        return seq_x, seq_y, np.zeros_like(seq_x), np.zeros_like(seq_y)



    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len - self.gap_len + 1

    def inverse_transform(self, data, scaler_type='y'):
        return self.scaler_y.inverse_transform(data) if scaler_type == 'y' else self.scaler_x.inverse_transform(data)


# ==============================================================================
# === KORAK 1: KONFIGURACIJA (SA NOVIM TCN PARAMETRIMA) ===
# ==============================================================================
class Configs:
    def __init__(self):
        # Parametri podataka i modela (MORAJU odgovarati podacima!)
        self.timeenc = 1
        self.root_path = '.'
        self.data_path = 'podaci_testiranje_spearman.csv'
        self.target = 'FLOW'
        self.features = 'M'
        self.freq = 'min'

        # Dimenzije sekvenci
        self.seq_len = 5
        self.label_len = 0
        self.gap_len = 0
        self.pred_len = 1

        # Broj karakteristika
        self.enc_in = 5
        self.c_out = 1

        # --- NOVI TCN HIPERPARAMETRI ---
        self.tcn_channels = [30] * 4  # [30, 30, 30, 30] -> 4 sloja sa po 30 kanala
        self.tcn_kernel_size = 5
        self.dropout = 0.2

        # Parametri treninga
        self.batch_size = 16
        self.learning_rate = 0.001  # TCN može da podnese malo višu stopu učenja
        self.num_epochs = 20
        self.use_gpu = True


configs = Configs()

# ==============================================================================
# === KORAK 2: PRIPREMA PODATAKA (Nepromenjeno!) ===
# ==============================================================================
print("Priprema podataka...")
train_dataset = Dataset_Custom(root_path=configs.root_path, data_path=configs.data_path, flag='train',
                               size=[configs.seq_len, configs.label_len, configs.pred_len], features=configs.features,
                               target=configs.target, freq=configs.freq, scale=True, timeenc=configs.timeenc,
                               gap_len=configs.gap_len)
val_dataset = Dataset_Custom(root_path=configs.root_path, data_path=configs.data_path, flag='val',
                             size=[configs.seq_len, configs.label_len, configs.pred_len], features=configs.features,
                             target=configs.target, freq=configs.freq, scale=True, timeenc=configs.timeenc,
                             gap_len=configs.gap_len)
test_dataset = Dataset_Custom(root_path=configs.root_path, data_path=configs.data_path, flag='test',
                              size=[configs.seq_len, configs.label_len, configs.pred_len], features=configs.features,
                              target=configs.target, freq=configs.freq, scale=True, timeenc=configs.timeenc,
                              gap_len=configs.gap_len)

train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=0, drop_last=False)
print("Podaci su spremni.")

# ==============================================================================
# === KORAK 3: INICIJALIZACIJA MODELA (Koristimo Model_TCN) ===
# ==============================================================================
device = torch.device("cuda:0" if configs.use_gpu and torch.cuda.is_available() else "cpu")
print(f"Koristi se uređaj: {device}")

# === KLJUČNA IZMENA: Inicijalizujemo TCN model umesto DLinear ===
model = Model_TCN(configs).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
print("\n--- TCN Model i komponente su spremni za trening. ---")

# ==============================================================================
# === KORAK 4: TRENIRANJE MODELA (Logika petlje je nepromenjena!) ===
# ==============================================================================
print("Početak treninga...")
train_start_time = time.time()
save_path = 'tcn_best_model_spearman.pth'  # Novi fajl za čuvanje TCN modela
early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)  # Možda smanjiti patience

for epoch in range(configs.num_epochs):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        outputs = model(batch_x)
        true_y = batch_y[:, -configs.pred_len:, :].to(device)
        loss = criterion(outputs, true_y)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            true_y = batch_y[:, -configs.pred_len:, :].to(device)
            loss = criterion(outputs, true_y)
            total_val_loss += loss.item()

    avg_train_loss, avg_val_loss = total_train_loss / len(train_loader), total_val_loss / len(val_loader)
    print(
        f"Epoha {epoch + 1}/{configs.num_epochs} | Vreme: {time.time() - epoch_start_time:.2f}s | Gubitak (Trening): {avg_train_loss:.7f} | Gubitak (Validacija): {avg_val_loss:.7f}")
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping - trening prekinut.")
        break

print(f"\nTrening završen. Ukupno vreme: {(time.time() - train_start_time) / 60:.2f} min")

# ==============================================================================
# === KORAK 5: EVALUACIJA (Koristimo Model_TCN) ===
# ==============================================================================
# === KLJUČNA IZMENA: Učitavamo u TCN arhitekturu ===
best_model = Model_TCN(configs).to(device)
best_model.load_state_dict(torch.load(save_path))
best_model.eval()
print("\nPočetak evaluacije na test setu sa NAJBOLJIM TCN modelom...")

preds, trues = [], []
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        outputs = best_model(batch_x)
        true_y = batch_y[:, -configs.pred_len:, :]
        preds.append(outputs.detach().cpu().numpy())
        trues.append(true_y.detach().cpu().numpy())

preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
preds_reshaped, trues_reshaped = preds.reshape(-1, configs.c_out), trues.reshape(-1, configs.c_out)
preds_inv, trues_inv = test_dataset.inverse_transform(preds_reshaped, 'y'), test_dataset.inverse_transform(
    trues_reshaped, 'y')

mse, mae, r2 = mean_squared_error(trues_inv, preds_inv), mean_absolute_error(trues_inv, preds_inv), r2_score(trues_inv,
                                                                                                             preds_inv)
print(f"Rezultati na test setu:\nMSE: {mse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")



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

# ==============================================================================
# === KORAK 6: VIZUELIZACIJA (Funkcija je nepromenjena!) ===
# ==============================================================================
'''
def analyze_prediction_at_date(start_date_str, dataset_for_scalers, model, configs, device):
    # Ova funkcija ostaje potpuno ista jer je apstraktna i radi sa bilo kojim modelom
    # koji poštuje definisani ulaz/izlaz.
    print(f"\n--- Analiza za specifičan datum: Početak istorije = {start_date_str} ---")
    df_full = pd.read_csv(os.path.join(configs.root_path, configs.data_path), decimal=',')
    df_full['TIMESTAMP'] = pd.to_datetime(df_full['TIMESTAMP'])
    df_full = df_full.set_index('TIMESTAMP').resample(configs.freq).mean().fillna(method='ffill').dropna().reset_index()

    target_start_date = pd.to_datetime(start_date_str)
    target_idx = df_full['TIMESTAMP'].searchsorted(target_start_date, side='left')

    if target_idx + configs.seq_len + configs.gap_len + configs.pred_len >= len(df_full):
        print(f"!!! GREŠKA: Nema dovoljno podataka nakon datuma {start_date_str} za analizu.")
        return

    input_df = df_full.iloc[target_idx: target_idx + configs.seq_len]
    true_future_df = df_full.iloc[
                     target_idx + configs.seq_len + configs.gap_len: target_idx + configs.seq_len + configs.gap_len + configs.pred_len]

    input_features_df = input_df.drop(columns=['TIMESTAMP', configs.target])
    input_scaled = dataset_for_scalers.scaler_x.transform(input_features_df.values)

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_scaled).unsqueeze(0).float().to(device)
        prediction_scaled = model(input_tensor).squeeze(0).cpu().numpy()

    predicted_future_inv = dataset_for_scalers.scaler_y.inverse_transform(prediction_scaled)

    # Priprema za plotovanje
    plot_df = df_full.iloc[target_idx: target_idx + configs.seq_len + configs.gap_len + configs.pred_len + 100]

    # Vizuelizacija
    plt.figure(figsize=(20, 8))
    plt.plot(plot_df['TIMESTAMP'], plot_df[configs.target], label=f'Stvarne vrednosti ({configs.target})', color='blue')
    plt.plot(true_future_df['TIMESTAMP'], predicted_future_inv.flatten(), label='Predviđene vrednosti (TCN)',
             color='red', linestyle='--', marker='o', markersize=4)
    if configs.gap_len > 0:
        gap_start_ts = input_df['TIMESTAMP'].iloc[-1]
        gap_end_ts = true_future_df['TIMESTAMP'].iloc[0]
        plt.axvspan(gap_start_ts, gap_end_ts, color='orange', alpha=0.2, label=f'Razmak ({configs.gap_len} min)')

    plt.title(f'Analiza TCN Predikcije sa Početkom Istorije na {start_date_str}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Poziv analize sa NAJBOLJIM TCN modelom
analyze_prediction_at_date(
    start_date_str='2025-02-01 00:00',
    dataset_for_scalers=train_dataset,
    model=best_model,
    configs=configs,
    device=device
)
'''
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
try:
    joblib.dump(train_dataset.scaler_x, 'scaler_x.pkl')
    joblib.dump(train_dataset.scaler_y, 'scaler_y.pkl')

except AttributeError:
    print("!!! GREŠKA: 'train_dataset' nema atribute 'scaler_x' ili 'scaler_y'.")