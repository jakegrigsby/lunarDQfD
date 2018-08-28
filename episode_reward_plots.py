import numpy as np
import matplotlib.pyplot as plt
from record_demonstrations import demo_avg, load_demo_data_from_file, reward_threshold_subset

class TrainingAnalyzer():
    """
    Framework for visualizing training data logged by TrainEpisodeLogger()
    """
    def __init__(self, filepath):
        try:
            with open(filepath, 'r') as f:
                self.processed_data = list()
                for line in f:
                    line.strip()
                    line.replace("\"",'')
                    data = line.split('\' \'') #cuts out loss metric strings
                    self.processed_data.append([(d.replace('\'','')) for d in data])
                    for value in self.processed_data:
                        if value == []:
                            self.processed_data.remove(value)
        except:
            raise FileNotFoundError()

        self.metrics = {
            'step':list(),
             'nb_steps':list(), 'episode':list(),
             'duration':list(), 'episode_steps':list(),
             'sps':list(), 'episode_reward':list(), 'reward_mean':list(),
              'reward_min':list(),'reward_max':list(),
             'action_mean':list(), 'action_min':list(), 'action_max':list(),
              'obs_mean':list(),
             'obs_min':list(), 'obs_max':list(), 'metrics_text':list()
            }

        for episode in self.processed_data[:-1]:
            if "WALL CLOCK TIME:" in episode[0]:
                continue
            self.metrics['step'].append(float(episode[16]))
            self.metrics['nb_steps'].append(float(episode[8]))
            self.metrics['episode'].append(float(episode[4]))
            self.metrics['duration'].append(float(episode[3]))
            self.metrics['episode_steps'].append(float(episode[6]))
            self.metrics['sps'].append(float(episode[15]))
            self.metrics['episode_reward'].append(float(episode[5]))
            self.metrics['reward_mean'].append(float(episode[13]))
            self.metrics['reward_min'].append(float(episode[14]))
            self.metrics['reward_max'].append(float(episode[12]))
            self.metrics['action_mean'].append(float(episode[1]))
            self.metrics['action_min'].append(float(episode[2]))
            self.metrics['action_max'].append(float(episode[0]))
            self.metrics['obs_mean'].append(float(episode[10]))
            self.metrics['obs_min'].append(float(episode[11]))
            self.metrics['obs_max'].append(float(episode[9]))
            self.metrics['metrics_text'].append(episode[7])

    def graph_metrics_by_episode(self, metric_list=[['episode_reward','-','b']], stylesheet='seaborn', smooth=True):
        """
        Metrics are passed in the form (metric_string_id, linetype, color) according to matplotlib conventions.
        """
        data = list()
        labels = list()
        colors = list()
        for metric in metric_list:
            if metric[0] not in self.metrics.keys():
                print(metric[0] + "Not a Valid Metric")
            else:
                data.append(self.metrics[metric[0]]) #appends the raw data
                labels.append(metric[0]) #appends the name of the metric
                colors.append(metric[-1])

        plt.style.use('fivethirtyeight')
        plt.xlabel('Episode')
        lines = list()
        for i, metric in enumerate(data):
            metric = self.savitzky_golay(metric, 197, 2)
            if i == 0:
                lines = plt.plot(self.metrics['episode'][:15000],metric[:15000],label=labels[i], color=colors[i])
            else:
                line = (plt.plot(self.metrics['episode'],metric,label=labels[i], color=colors[i]))
                lines.append(line[0])

        if len(labels) == 1:
            plt.ylabel(labels[0])
        else:
            plt.legend([line for line in lines],[label for label in labels])


    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        """
        Noise-smoothing function for charts and graphs. From SciPy documentation.
        """
        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( np.array(y[1:half_window+1][::-1]) - np.array(y[0]) )
        lastvals = y[-1] + np.abs( np.array(y[-half_window-1:-1][::-1]) - np.array(y[-1]))
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

#DQN vs DQfD
e = TrainingAnalyzer('./model_saves/expert_lander_REWARD_DATA.txt')
e.graph_metrics_by_episode(metric_list=[['episode_reward','-','grey']])
e = TrainingAnalyzer('./model_saves/student_lander15k_REWARD_DATA.txt')
e.graph_metrics_by_episode(metric_list=[['episode_reward','-','purple']])
demos = reward_threshold_subset(load_demo_data_from_file('./model_saves/demos.npy'), 0)
demo_avg = demo_avg(demos)
plt.plot([i for i in range(15000)],[demo_avg for i in range(15000)],color='blue')
plt.legend(['PDD','DQfD','Demo Avg'])
plt.show()

# #Pretraining Lengths
# e = TrainingAnalyzer('./model_saves/student_lander15k_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','grey']])
# e = TrainingAnalyzer('./model_saves/student_lander50k_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','#cce8c9']])
# e = TrainingAnalyzer('./model_saves/student_lander100k_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','#ebf298']])
# e = TrainingAnalyzer('./model_saves/student_lander300k_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','purple']])
# demos = reward_threshold_subset(load_demo_data_from_file('./model_saves/demos.npy'), 0)
# demo_avg = demo_avg(demos)
# plt.plot([i for i in range(15000)],[demo_avg for i in range(15000)],color='blue')
# plt.legend(['15k','50k','100k','300k','Demo Avg'])
# plt.show()

#l2 regularization
# e = TrainingAnalyzer('./model_saves/student_lander300k_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','#d6bdef']])
# e = TrainingAnalyzer('./model_saves/student_lander300k001_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','#8962af']])
# e = TrainingAnalyzer('./model_saves/student_lander300k01_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','#371d51']])
# demos = reward_threshold_subset(load_demo_data_from_file('./model_saves/demos.npy'), 0)
# demo_avg = demo_avg(demos)
# plt.plot([i for i in range(15000)],[demo_avg for i in range(15000)],color='blue',label='Demonstration min.')
# plt.legend(['.0001','.001','.01','Demo Avg'])
# plt.show()

# #expert only
# e = TrainingAnalyzer('./model_saves/expert_lander_REWARD_DATA.txt')
# e.graph_metrics_by_episode(metric_list=[['episode_reward','-','grey']])
# plt.legend(['PDD'])
# plt.show()
