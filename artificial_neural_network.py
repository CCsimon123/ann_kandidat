import matplotlib.colors
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from numpy import linspace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import cm
from scipy.interpolate import interpn
from scipy.stats import kde, gaussian_kde
from plot_fun import *

# Created by Simon Carlson April 2022

# Flags
create_new_data = False
train_data_frac = 0.9
load_model = True
save_model = True
show_histogram = False
show_val_acc = False
show_train_acc = False
get_score = True

num_of_input = 3
batch_size = 5000
learning_rate = 0.00005
number_of_epochs = 0

path_to_save_model_to = r'saved_models\final_model_azm_no_feature3.pth'
path_to_load_from = r'saved_models\final_model_azm_no_feature3.pth'
path_to_data = r'data\params_inc_azi.csv'
train_path = Path(r'data\train_data.csv')
val_path = Path(r'data\val_data.csv')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')


if create_new_data:
    df = pd.read_csv(path_to_data, names=['SWH', 'mean', 'var', 'azm', 'flag'])
    # Removes null values
    df.drop(df[df['SWH'].isnull()].index, inplace=True)
    rng = RandomState(4)

    df['mean'] = (df['mean'] - df['mean'].mean()) / (df['mean'].std())
    df['var'] = (df['var'] - df['var'].mean()) / (df['var'].std())
    df['azm'] = (df['azm'] - df['azm'].mean()) / (df['azm'].std())

    #df = df[['mean', 'var', 'SWH']]
    df = df[['mean', 'var', 'azm', 'SWH']]

    train = df.sample(frac=train_data_frac, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(val_path, index=False)


class CustomCsvDataset():
    def __init__(self, dataset):
        self.dataset = dataset.to(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_data = self.dataset[idx, 0:(self.dataset.size(1) - 1)]
        label = self.dataset[idx, self.dataset.size(1) - 1]
        return input_data, label


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(num_of_input, 30)
        #self.drop = torch.nn.Dropout(0.5) # add dropout if the model starts to overfit
        self.hid2 = torch.nn.Linear(30, 30)
        self.hid3 = torch.nn.Linear(30, 30)
        self.hid4 = torch.nn.Linear(30, 30)
        self.hid5 = torch.nn.Linear(30, 30)
        self.hid6 = torch.nn.Linear(30, 30)
        self.hid7 = torch.nn.Linear(30, 30)
        self.hid8 = torch.nn.Linear(30, 30)

        self.output = torch.nn.Linear(30, 1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid4(z))
        z = torch.relu(self.hid5(z))
        z = torch.relu(self.hid6(z))
        z = torch.relu(self.hid7(z))
        z = torch.relu(self.hid8(z))
        '''
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid2(z))
        '''
        z = self.output(z)
        return z


def main():
    torch.manual_seed(4)
    np.random.seed(4)

    training_data = np.loadtxt(train_path, dtype=np.float32, delimiter=",", skiprows=1)
    if show_histogram:
        max_SWH = 3
        bin = np.linspace(0, max_SWH, 100)
        plt.hist(training_data[:, num_of_input], bins=bin)
        plt.show()

    training_data = torch.from_numpy(training_data)
    train_data = CustomCsvDataset(training_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = np.loadtxt(val_path, dtype=np.float32, delimiter=",", skiprows=1)
    validation_data = torch.from_numpy(validation_data)
    val_data = CustomCsvDataset(validation_data)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    net = Net().to(device)
    # loads the old model
    if load_model:
        net.load_state_dict(torch.load(path_to_load_from, map_location=device))

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    net.train()

    start_time = time.time()
    for epoch in range(number_of_epochs):
        torch.manual_seed(1+epoch) # recovery reproducibility
        epoch_loss = 0.0

        for (idx, X) in enumerate(train_dataloader):
            (input_data, label) = X
            optimizer.zero_grad()
            output = net(input_data)
            output = torch.squeeze(output)

            loss_val = loss_func(output, label)
            epoch_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == number_of_epochs-1:
            print(f'epoch {epoch} loss = {epoch_loss}')

    end_time = time.time()
    print(f'time for the training: {end_time - start_time}')

    def accuracy(model, ds, ok_error):
        correct = 0
        total = 0
        for val_data, label in ds:
            with torch.no_grad():
                output = model(val_data)
            abs_delta = np.abs(output.item()-label.item())
            if abs_delta < ok_error:
                correct += 1
            else:
                pass
                #print(f'got it wrong: val_data {val_data} label {label}, guessed {output}')
            total += 1
        acc = correct/total
        return acc

    net.eval()
    ok_error = 0.2
    if show_train_acc:
        train_acc = accuracy(net, train_data, ok_error)
        print(f'train accuracy: {train_acc}')
    if show_val_acc:
        val_acc = accuracy(net, val_data, ok_error)
        print(f'validation ({ok_error}m) accuracy: {val_acc}')

        ok_error = 0.6
        val_acc = accuracy(net, val_data, ok_error)
        print(f'validation ({ok_error}m) accuracy: {val_acc}')

    if save_model:
        torch.save(net.state_dict(), path_to_save_model_to)

    def get_model_guess_vs_target_arrs(model, ds):
        model_guess_arr = np.empty(len(ds))
        targel_arr = np.empty(len(ds))

        for i, data in enumerate(ds):
            val_data, label = data
            with torch.no_grad():
                output = model(val_data)
            targel_arr[i] = label
            model_guess_arr[i] = output

        return targel_arr, model_guess_arr

    if get_score:
        target_arr, model_guess_arr = get_model_guess_vs_target_arrs(net, val_data)
        # Calculates RMSE, R-value and MAE
        mse = mean_squared_error(target_arr, model_guess_arr)
        rmse = np.sqrt(mse)
        print(f"RMSE for full validation set: {rmse:.3f}")

        res = r2_score(target_arr, model_guess_arr)
        res = np.sqrt(res)
        print(f"R-value for full validation set: {res:.3f}")

        mean_absolute_error = np.mean(np.abs(target_arr - model_guess_arr))
        print(f"Mean absolute error: {mean_absolute_error:.3f}")

        bias_error = np.mean(target_arr - model_guess_arr)
        print(f"bias error: {bias_error:.3f}")

        plot_hist2d(target_arr, model_guess_arr, bins=40)

if __name__ == '__main__':
    main()
    plt.show()
