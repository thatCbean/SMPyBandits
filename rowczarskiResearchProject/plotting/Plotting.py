import os
import time
from datetime import datetime

PLOT_DIR = 'plots'
SEMILOG_X = False
SEMILOG_Y = False
LOG_LOG = False


def create_folder_plot_if_needed():
    if os.path.isdir(PLOT_DIR):
        print("{}/ is already a directory here...".format(PLOT_DIR))
    elif os.path.isfile(PLOT_DIR):
        raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(PLOT_DIR))
    else:
        os.mkdir(PLOT_DIR)


class Plotting:
    def __init__(self, evaluator, configuration, saveAllFigures=False):
        self.saveFigure = None
        self.mainFigure = None
        self.imageName = None
        self.plot_dir = None
        self.evaluator = evaluator
        self.configuration = configuration
        self.saveAllFigures = saveAllFigures
        create_folder_plot_if_needed()

    def create_subfolder(self, N, environment, environmentId, hashValue):
        subfolder = "SP__K{}_T{}_N{}__{}_algos_{}".format(environment.nbArms, self.configuration['horizon'],
                                                       self.configuration['repetitions'],
                                                       len(self.configuration['policies']),
                                                          datetime.now().strftime("%Y%m%d%H%M"))
        self.plot_dir = os.path.join(PLOT_DIR, subfolder)
        # Get the name of the output file
        self.imageName = "main____env{}-{}_{}".format(environmentId + 1, N, hashValue)
        self.mainFigure = os.path.join(self.plot_dir, self.imageName)
        self.saveFigure = self.mainFigure

        # Create the sub folder
        if os.path.isdir(self.plot_dir):
            print("{} is already a directory here...".format(self.plot_dir))
        elif os.path.isfile(self.plot_dir):
            raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(self.plot_dir))
        else:
            os.mkdir(self.plot_dir)

    def plot_history_of_means(self, envId):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_HistoryOfMeans')
            print(" - Plotting the history of means, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotHistoryOfMeans(envId, savefig=savefig)  # XXX To save the figure
        else:
            self.evaluator.plotHistoryOfMeans(envId)  # XXX To plot without saving

    def plot_boxplot_regret(self, envId):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_BoxPlotRegret')
            print(" - Plotting the boxplot of last regrets, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotLastRegrets(envId, boxplot=True, savefig=savefig)  # XXX To save the figure
        else:
            self.evaluator.plotLastRegrets(envId, boxplot=True)

    def plot_running_times(self, envId):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_RunningTimes')
            print(" - Plotting the running times, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotRunningTimes(envId, savefig=savefig)  # XXX To save the figure
        else:
            self.evaluator.plotRunningTimes(envId)

    def plot_memory_consumption(self, envId):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_MemoryConsumption')
            print(" - Plotting the memory consumption, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotMemoryConsumption(envId, savefig=savefig)  # XXX To save the figure
        else:
            self.evaluator.plotMemoryConsumption(envId)

    def plot_number_of_cp_detections(self, envId):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_NumberOfCPDetections')
            print(" - Plotting the number of detected change-points, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotNumberOfCPDetections(envId, savefig=savefig)  # XXX To save the figure
        else:
            self.evaluator.plotNumberOfCPDetections(envId)

    def plot_mean_rewards(self, envId, semilogx, semilogy, loglog):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_MeanRewards')
            print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                                       meanReward=True)  # XXX To save the figure
        else:
            self.evaluator.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog, meanReward=True)

    def plot_all_regrets(self, envId, semilogx, semilogy, loglog):
        if self.saveAllFigures:
            savefig = self.mainFigure
            print(" - Plotting the cumulative rewards, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotRegrets(envId, savefig=savefig, moreAccurate=True)
            savefig = self.mainFigure.replace('main', 'main_LessAccurate')
            self.evaluator.plotRegrets(envId, savefig=savefig, moreAccurate=False)
            savefig = self.mainFigure.replace('main', 'main_BestArmPulls')
            print(" - Plotting the probability of picking the best arm, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotBestArmPulls(envId, savefig=savefig)
            savefig = self.mainFigure.replace('main', 'main_semilogy')
            self.evaluator.plotRegrets(envId, savefig=savefig, semilogy=True)

            if self.configuration['horizon'] >= 1000:
                savefig = self.mainFigure.replace('main', 'main_loglog')
                self.evaluator.plotRegrets(envId, savefig=savefig, loglog=True)

            if self.configuration['repetitions'] > 1:
                savefig = savefig.replace('main', 'main_STD')
                # self.evaluator.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                #                            plotSTD=True) TODO: Fix this
                savefig = savefig.replace('main', 'main_MaxMin')
                self.evaluator.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                                           plotMaxMin=True)

        else:
            self.evaluator.plotRegrets(envId, moreAccurate=True)
            self.evaluator.plotRegrets(envId, moreAccurate=False)
            self.evaluator.plotBestArmPulls(envId)
            self.evaluator.plotRegrets(envId, semilogy=True)
            if self.configuration['horizon'] >= 1000:
                self.evaluator.plotRegrets(envId, loglog=True)
            if self.configuration['repetitions'] > 1:
                # self.evaluator.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                #                            plotSTD=True) TODO: Fix this
                self.evaluator.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                                           plotMaxMin=True)

    def plot_normalized_regrets(self, envId, semilogx, semilogy, loglog):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_Normalized')
            print(" - Plotting the mean rewards, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                                       normalizedRegret=True)
            if self.configuration['repetitions'] > 1:
                savefig = savefig.replace('main', 'main_STD')
                # self.evaluator.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy,
                #                            loglog=loglog, normalizedRegret=True, plotSTD=True) TODO: Fix this

                savefig = savefig.replace('main', 'main_MaxMin')
                self.evaluator.plotRegrets(envId, savefig=savefig, semilogx=semilogx, semilogy=semilogy,
                                           loglog=loglog, normalizedRegret=True, plotMaxMin=True)
        else:
            self.evaluator.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                                       normalizedRegret=True)
            if self.configuration['repetitions'] > 1:
                # self.evaluator.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                #                            normalizedRegret=True, plotSTD=True) TODO: Fix this
                self.evaluator.plotRegrets(envId, semilogx=semilogx, semilogy=semilogy, loglog=loglog,
                                           normalizedRegret=True, plotMaxMin=True)

    def plot_histograms_regret(self, envId):
        if self.saveAllFigures:
            savefig = self.mainFigure.replace('main', 'main_HistogramsRegret')
            print(" - Plotting the histograms of regrets, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotLastRegrets(envId, savefig=savefig)
            for sharex, sharey in [(True, False)]:
                savefig = self.mainFigure.replace('main', 'main_HistogramsRegret{}{}'.format(
                    "_shareX" if sharex else "",
                    "_shareY" if sharey else "",
                ))
                print(" - Plotting the histograms of regrets, and saving the plot to {} ...".format(savefig))
                self.evaluator.plotLastRegrets(envId, savefig=savefig, sharex=sharex, sharey=sharey)
            print(" - Plotting the histograms of regrets for each algorithm separately, and saving the plots ...")
            savefig = self.mainFigure.replace('main', 'main_HistogramsRegret')
            print(" - Plotting the histograms of regrets, and saving the plot to {} ...".format(savefig))
            self.evaluator.plotLastRegrets(envId, all_on_separate_figures=True, savefig=savefig)
        else:
            self.evaluator.plotLastRegrets(envId)
            for sharex, sharey in [(True, False)]:
                self.evaluator.plotLastRegrets(envId, sharex=sharex, sharey=sharey)

    def plot_all(self, envId):
        self.plot_history_of_means(envId)
        self.plot_boxplot_regret(envId)
        self.plot_running_times(envId)
        self.plot_memory_consumption(envId)
        self.plot_number_of_cp_detections(envId)
        self.plot_mean_rewards(envId, semilogx=SEMILOG_X, semilogy=SEMILOG_Y, loglog=LOG_LOG)
        self.plot_all_regrets(envId, semilogx=SEMILOG_X, semilogy=SEMILOG_Y, loglog=LOG_LOG)
        self.plot_normalized_regrets(envId, semilogx=SEMILOG_X, semilogy=SEMILOG_Y, loglog=LOG_LOG)
        self.plot_histograms_regret(envId)
