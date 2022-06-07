import matplotlib.pyplot as plt

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("Value")
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)

def plot_ts_datasets(dataset1=[], dataset2=[], label1="ds1", label2="ds2", log=False):
    plt.plot(dataset1, label=label1)
    plt.plot(dataset2, label=label2)
    if log:
      plt.yscale('log')
    plt.xlabel('Frequency bin')
    plt.ylabel('Difference')
    plt.legend()
    plt.show()