#!/usr/bin/env python
# title             :training_pytorch.py
# description       :Training deep learning for distinguishing viruses from hosts (PyTorch version)
# author            :Assistant (adapted to match original script outputs)
# date              :20251029
# version           :1.4
# usage             :python training_pytorch.py -l 1000 -i ./train_example/tr/encode -j ./train_example/val/encode -o ./train_example/models -f 10 -n 1000 -d 1000 -e 50
# required packages :numpy, torch, scikit-learn
#==============================================================================

import numpy as np
import os
import sys
import random
import optparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

channel_num = 4
prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-l", "--len", action="store", type=int, dest="contigLength",
                  help="contig Length")
parser.add_option("-i", "--intr", action="store", type="string", dest="inDirTr",
                  default='./', help="input directory for training data")
parser.add_option("-j", "--inval", action="store", type="string", dest="inDirVal",
                  default='./', help="input directory for validation data")
parser.add_option("-o", "--out", action="store", type="string", dest="outDir",
                  default='./', help="output directory")
parser.add_option("-f", "--fLen1", action="store", type=int, dest="filter_len1",
                  help="the length of filter")
parser.add_option("-n", "--fNum1", action="store", type=int, dest="nb_filter1",
                  default=0, help="number of filters in the convolutional layer")
parser.add_option("-d", "--dense", action="store", type=int, dest="nb_dense",
                  default=0, help="number of neurons in the dense layer")
parser.add_option("-e", "--epochs", action="store", type=int, dest="epochs",
                  default=0, help="number of epochs")
parser.add_option("-b", "--batch", action="store", type=int, dest="batch_size",
                  default=None, help="batch size (optional; default auto)")

(options, args) = parser.parse_args()
if (options.contigLength is None or
        options.filter_len1 is None or
        options.nb_filter1 is None or options.nb_dense is None):
    sys.stderr.write(prog_base + ": ERROR: missing required command-line argument\n")
    parser.print_help()
    sys.exit(0)

contigLength = options.contigLength
filter_len1 = options.filter_len1
nb_filter1 = options.nb_filter1
nb_dense = options.nb_dense
inDirTr = options.inDirTr
inDirVal = options.inDirVal
outDir = options.outDir
if not os.path.exists(outDir):
    os.makedirs(outDir)
epochs = options.epochs
batch_size_cli = options.batch_size

# fixed seed (no CLI option as requested)
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

contigLengthk = contigLength / 1000
if isinstance(contigLengthk, float) and contigLengthk.is_integer():
    contigLengthk = int(contigLengthk)
contigLengthk = str(contigLengthk)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Model (matches the Keras architecture)
# ---------------------------
class SiameseModel(nn.Module):
    def __init__(self, filter_len1, nb_filter1, nb_dense, dropout_pool=0.1, dropout_dense=0.1):
        super(SiameseModel, self).__init__()
        # Conv1d expects (batch, channels, length) -> we will permute inputs
        self.conv1d = nn.Conv1d(in_channels=channel_num, out_channels=nb_filter1, kernel_size=filter_len1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_pool = nn.Dropout(dropout_pool)
        self.dense1 = nn.Linear(nb_filter1, nb_dense)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.dense2 = nn.Linear(nb_dense, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_branch(self, x):
        # x shape: (batch, seq_len, channel_num)
        x = x.permute(0, 2, 1)  # -> (batch, channel_num, seq_len)
        x = self.conv1d(x)
        x = self.global_max_pool(x).squeeze(-1)  # -> (batch, nb_filter1)
        x = self.dropout_pool(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout_dense(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x_fw, x_bw):
        out_fw = self.forward_branch(x_fw)
        out_bw = self.forward_branch(x_bw)
        return (out_fw + out_bw) / 2.0  # (batch, 1)


# ---------------------------
# Helper: find single file by pattern (like original script)
# ---------------------------
def find_one_file_with_keywords(indir, must_have_substr_list, contains_substr):
    """
    Find first file in indir that contains all substrings in must_have_substr_list and contains_substr.
    Returns filename (basename) or raises.
    """
    files = sorted(os.listdir(indir))
    for fn in files:
        fl = fn.lower()
        ok = all((s.lower() in fl) for s in must_have_substr_list)
        if ok and (contains_substr.lower() in fl):
            return fn
    raise FileNotFoundError(f"No file found in {indir} matching keys {must_have_substr_list} and containing '{contains_substr}'")

# ---------------------------
# Load data (mimic Keras script outputs)
# ---------------------------
print("...loading data...")

# virus (phage) data
print("...loading virus data...")
try:
    filename_codetrfw = [x for x in os.listdir(inDirTr) if 'codefw.npy' in x.lower() and 'virus' in x.lower() and (contigLengthk + 'k') in x.lower()][0]
    print("data for training " + filename_codetrfw)
    phageRef_codetrfw = np.load(os.path.join(inDirTr, filename_codetrfw))
    phageRef_codetrbw = np.load(os.path.join(inDirTr, filename_codetrfw.replace('fw', 'bw')))
except Exception as e:
    print("ERROR: could not load virus training files:", e)
    sys.exit(1)

try:
    filename_codevalfw = [x for x in os.listdir(inDirVal) if 'codefw.npy' in x.lower() and 'virus' in x.lower() and (contigLengthk + 'k') in x.lower()][0]
    print("data for validation " + filename_codevalfw)
    phageRef_codevalfw = np.load(os.path.join(inDirVal, filename_codevalfw))
    phageRef_codevalbw = np.load(os.path.join(inDirVal, filename_codevalfw.replace('fw', 'bw')))
except Exception as e:
    print("ERROR: could not load virus validation files:", e)
    sys.exit(1)

# host data
print("...loading host data...")
try:
    filename_codetrfw = [x for x in os.listdir(inDirTr) if 'codefw.npy' in x.lower() and 'host' in x.lower() and (contigLengthk + 'k') in x.lower()][0]
    print("data for training " + filename_codetrfw)
    hostRef_codetrfw = np.load(os.path.join(inDirTr, filename_codetrfw))
    hostRef_codetrbw = np.load(os.path.join(inDirTr, filename_codetrfw.replace('fw', 'bw')))
except Exception as e:
    print("ERROR: could not load host training files:", e)
    sys.exit(1)

try:
    filename_codevalfw = [x for x in os.listdir(inDirVal) if 'codefw.npy' in x.lower() and 'host' in x.lower() and (contigLengthk + 'k') in x.lower()][0]
    print("data for validation " + filename_codevalfw)
    hostRef_codevalfw = np.load(os.path.join(inDirVal, filename_codevalfw))
    hostRef_codevalbw = np.load(os.path.join(inDirVal, filename_codevalfw.replace('fw', 'bw')))
except Exception as e:
    print("ERROR: could not load host validation files:", e)
    sys.exit(1)

# ---------------------------
# Combine and shuffle (as original)
# ---------------------------
print("...combining V and H...")
# training
Y_tr = np.concatenate((np.repeat(0, hostRef_codetrfw.shape[0]), np.repeat(1, phageRef_codetrfw.shape[0])))
X_trfw = np.concatenate((hostRef_codetrfw, phageRef_codetrfw), axis=0)
del hostRef_codetrfw, phageRef_codetrfw
X_trbw = np.concatenate((hostRef_codetrbw, phageRef_codetrbw), axis=0)
del hostRef_codetrbw, phageRef_codetrbw

print("...shuffling training data...")
index_trfw = list(range(0, X_trfw.shape[0]))
np.random.shuffle(index_trfw)
X_trfw_shuf = X_trfw[np.ix_(index_trfw, range(X_trfw.shape[1]), range(X_trfw.shape[2]))]
del X_trfw
X_trbw_shuf = X_trbw[np.ix_(index_trfw, range(X_trbw.shape[1]), range(X_trbw.shape[2]))]
del X_trbw
Y_tr_shuf = Y_tr[index_trfw]

# validation combine
Y_val = np.concatenate((np.repeat(0, hostRef_codevalfw.shape[0]), np.repeat(1, phageRef_codevalfw.shape[0])))
X_valfw = np.concatenate((hostRef_codevalfw, phageRef_codevalfw), axis=0)
del hostRef_codevalfw, phageRef_codevalfw
X_valbw = np.concatenate((hostRef_codevalbw, phageRef_codevalbw), axis=0)
del hostRef_codevalbw, phageRef_codevalbw

# ---------------------------
# Determine batch size (approx same as original)
# ---------------------------
if batch_size_cli is None:
    # mimic original: batch_size=int(X_trfw_shuf.shape[0]/(1000*1000/contigLength))
    denom = (1000 * 1000 / contigLength) if contigLength != 0 else 1
    batch_size = max(1, int(X_trfw_shuf.shape[0] / denom))
else:
    batch_size = batch_size_cli


# Convert to tensors and dataloaders
X_trfw_t = torch.from_numpy(X_trfw_shuf).float()
X_trbw_t = torch.from_numpy(X_trbw_shuf).float()
Y_tr_t = torch.from_numpy(Y_tr_shuf).float().unsqueeze(1)

X_valfw_t = torch.from_numpy(X_valfw).float()
X_valbw_t = torch.from_numpy(X_valbw).float()
Y_val_t = torch.from_numpy(Y_val).float().unsqueeze(1)

train_dataset = TensorDataset(X_trfw_t, X_trbw_t, Y_tr_t)
val_dataset = TensorDataset(X_valfw_t, X_valbw_t, Y_val_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Model init / load if exists
# ---------------------------
modPattern = 'model_siamese_varlen_' + contigLengthk + 'k_fl' + str(filter_len1) + '_fn' + str(nb_filter1) + '_dn' + str(nb_dense)
modName = os.path.join(outDir, modPattern + '.pth')

print("...building model...")
model = SiameseModel(filter_len1, nb_filter1, nb_dense).to(device)

if os.path.isfile(modName):
    try:
        model.load_state_dict(torch.load(modName, map_location=device))
        print("...model exists...")
    except Exception as e:
        print("Warning: failed to load existing model, will train new one:", e)

# loss & optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# Training loop with checkpoint (save best by val_loss) and early stopping (monitor val_loss)
# ---------------------------
print("...fitting model...")
print(contigLengthk + 'k_fl' + str(filter_len1) + '_fn' + str(nb_filter1) + '_dn' + str(nb_dense) + '_ep' + str(epochs))

best_val_loss = np.inf
patience = 5
no_improve_epochs = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for Xfw_batch, Xbw_batch, Y_batch in train_loader:
        Xfw_batch = Xfw_batch.to(device)
        Xbw_batch = Xbw_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(Xfw_batch, Xbw_batch)  # (batch,1)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / (n_batches if n_batches > 0 else 1)

    # compute validation loss and AUC
    model.eval()
    y_trues = []
    y_scores = []
    val_loss_accum = 0.0
    val_samples = 0
    with torch.no_grad():
        for Xfw_batch, Xbw_batch, Y_batch in val_loader:
            Xfw_batch = Xfw_batch.to(device)
            Xbw_batch = Xbw_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(Xfw_batch, Xbw_batch)  # (batch,1)
            # loss for this batch (sum over batch)
            batch_loss = criterion(outputs, Y_batch).item()
            val_loss_accum += batch_loss * Xfw_batch.shape[0]  # multiply by batch size to accumulate per-sample
            val_samples += Xfw_batch.shape[0]
            y_scores.append(outputs.cpu().numpy().ravel())
            y_trues.append(Y_batch.cpu().numpy().ravel())
    if val_samples > 0:
        val_loss = val_loss_accum / val_samples
    else:
        val_loss = float('nan')

    # compute val auc for logging
    if len(y_scores) > 0:
        y_scores_all = np.concatenate(y_scores, axis=0)
        y_trues_all = np.concatenate(y_trues, axis=0)
        try:
            val_auc = roc_auc_score(y_trues_all, y_scores_all)
        except Exception:
            val_auc = float('nan')
    else:
        val_auc = float('nan')

    # print epoch summary (train loss and val loss/auc)
    print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.6f}  val_loss: {val_loss:.6f}  val_auc: {val_auc:.6f}")

    # checkpoint logic: save best by val_loss (smaller is better) -- matches Keras default
    if np.isfinite(val_loss) and val_loss < best_val_loss:
        prev = best_val_loss if np.isfinite(best_val_loss) else float('nan')
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, modName)

        print(f"Epoch {epoch+1}: val_loss improved from {prev:.6f} to {val_loss:.6f}, saving model to {modName}")
    else:
        no_improve_epochs += 1

    # early stopping based on val_loss (matches the checkpoint metric)
    if no_improve_epochs >= patience:
        print(f"Early stopping: no improvement in val_loss for {patience} epochs.")
        break

# ---------------------------
# Final evaluation & prediction outputs (mimic original)
# ---------------------------
# load best model if saved
if os.path.isfile(modName):
    try:
        model.load_state_dict(torch.load(modName, map_location=device))
        print("Loaded best model from disk for final evaluation.")
    except Exception:
        pass

# predict on train set
print("...predicting tr...\n")
model.eval()
y_scores_tr = []
with torch.no_grad():
    for i in range(X_trfw_shuf.shape[0]):
        xfw = X_trfw_shuf[i:i+1]
        xbw = X_trbw_shuf[i:i+1]
        t_xfw = torch.from_numpy(xfw).float().to(device)
        t_xbw = torch.from_numpy(xbw).float().to(device)
        out = model(t_xfw, t_xbw)
        y_scores_tr.append(out.cpu().numpy().ravel())
y_scores_tr = np.concatenate(y_scores_tr, axis=0)
try:
    auc_tr = roc_auc_score(Y_tr_shuf, y_scores_tr)
    print('auc_tr=' + str(auc_tr) + '\n')
except Exception as e:
    print("Could not compute auc_tr:", e)

# optionally save train preds (commented in original, but we keep as commented)
# np.savetxt(os.path.join(outDir, modPattern + '_trfw_Y_pred.txt'), np.transpose(y_scores_tr))
# np.savetxt(os.path.join(outDir, modPattern + '_trfw_Y_true.txt'), np.transpose(Y_tr_shuf))

# predict on val set
print("...predicting val...\n")
y_scores_val = []
with torch.no_grad():
    for i in range(X_valfw.shape[0]):
        xfw = X_valfw[i:i+1]
        xbw = X_valbw[i:i+1]
        t_xfw = torch.from_numpy(xfw).float().to(device)
        t_xbw = torch.from_numpy(xbw).float().to(device)
        out = model(t_xfw, t_xbw)
        y_scores_val.append(out.cpu().numpy().ravel())
y_scores_val = np.concatenate(y_scores_val, axis=0)
try:
    auc_val = roc_auc_score(Y_val, y_scores_val)
    print('auc_val=' + str(auc_val) + '\n')
    # save predictions and true as in original
    np.savetxt(os.path.join(outDir, modPattern + '_valfw_Y_pred.txt'), np.transpose(y_scores_val))
    np.savetxt(os.path.join(outDir, modPattern + '_valfw_Y_true.txt'), np.transpose(Y_val))
except Exception as e:
    print("Could not compute auc_val or save predictions:", e)

# save final model if not already saved
if not os.path.isfile(modName):
    torch.save(model.state_dict(), modName)
    print(f"Model saved to {modName}")
else:
    print(f"Model (best) is at {modName}")
