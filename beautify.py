import argparse
# import tensorflow as tf
# from tensorflow.python.summary import event_accumulator as ea
from tensorboard.backend.event_processing import event_accumulator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper")

def plot(params):
  ''' beautify tf log
      Use better library (seaborn) to plot tf event file'''

  log_path = params['logdir']
  smooth_space = params['smooth']
  color_code = params['color']

  acc = ea.EventAccumulator(log_path)
  acc.Reload()

  # only support scalar now
  scalar_list = acc.Tags()['scalars']
  print(scalar_list)

  x_list = []
  y_list = []
  x_list_raw = []
  y_list_raw = []
  for tag in scalar_list:
    
    if tag != 'Return1000':
      continue

    x = [int(s.step) for s in acc.Scalars(tag)]
    y = [s.value for s in acc.Scalars(tag)]

    # segmentation
    idx = []
    for i, v in enumerate(x):
      if v == 2048:
        idx.append(i)

    curve_list_x = []
    curve_list_y = []
    for i in range(len(idx)-1):
      curve_list_x.append(x[idx[i] : idx[i+1]-1])
      curve_list_y.append(y[idx[i] : idx[i+1]-1])
    curve_list_x.append(x[idx[-1] : -1])
    curve_list_y.append(y[idx[-1] : -1]) #raw curve

    

    # smooth curve
    curve_list_x_ = []
    curve_list_y_ = []
    for j in range(0, len(curve_list_x)):
      x_ = []
      y_ = []
      for i in range(0, len(curve_list_x[j]), smooth_space):
        x_.append(curve_list_x[j][i])
        y_.append(sum(curve_list_y[j][i:i+smooth_space]) / float(smooth_space))    
      x_.append(curve_list_x[j][-1])
      y_.append(curve_list_x[j][-1])
      curve_list_x_.append(x_)
      curve_list_y_.append(y_)


  for i in range(len(curve_list_x)):
    plt.figure(1)
    plt.subplot(111)
    plt.title(tag)  
    plt.plot(curve_list_x[i], curve_list_y[i], color=colors.to_rgba(color_code, alpha=0.4))
    plt.plot(curve_list_x_[i], curve_list_y_[i], color=color_code, linewidth=1.5)
  plt.show()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', default='./logdir', type=str, help='logdir to event file')
  parser.add_argument('--smooth', default=100, type=float, help='window size for average smoothing')
  parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  plot(params)
