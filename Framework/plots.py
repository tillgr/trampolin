import pandas as pd
import numpy as np
import shap
from holoviews.plotting.util import process_cmap
import matplotlib.pyplot as plt
import sklearn
from matplotlib.colors import ListedColormap


"""
#                                       #
#                                       #
#             for shap plots            #
#          and confusionmatrix          #
#                                       #
"""


def create_colormap(shap_y_test, shap_values=None):
    """
    Takes values and creates unique colormap, so that each jump, have always the sam color.
    Function is needed in bar_plots for 'single' and 'summary_color' bar plots
    -> contains color dict for all jumps, they used till now, so must be updated if more jumps avaible.
    An overview of the colors can be found in cmap.pdf
    :param shap_values: output from explainer.shap_values(shap_x_test); If shap_values are None, must shap_y_test a string.
    :param shap_y_test: sampled y_test or single string jump
    :return: colormap or single color if shap_values=None
    """

    color_dict = {
        '1 3/4 Salto vw B': '#843c39',
        '1 3/4 Salto vw C': '#ad494a',
        '3/4 Salto rw A': '#d6616b',
        '3/4 Salto vw A': '#e7969c',
        'Barani A': '#e6550d',
        'Barani B': '#fd8d3c',
        'Barani C': '#fdae6b',
        'Bauchsprung': '#544000',
        'Bücksprung': '#8c6d31',
        'Grätschwinkel': '#bd9e39',
        'Hocksprung': '#e7ba52',
        'Von Bauch in Stand': '#ffe07d',
        'Strecksprung': '#e7cb94',
        'Fliffis B': '#7b4173',
        'Fliffis C': '#a55194',
        'Baby- Fliffis C': '#ce6dbd',
        'Fliffis aus B': '#d9599c',
        'Fliffis aus C': '#f781bf',
        'Fliffis- Rudi B': '#ff8cdb',
        'Fliffis- Rudi C': '#ffb3e7',
        'Rudi': '#c91a93',
        'Halb ein Triffis C': '#756bb1',
        'Triffis B': '#9e9ac8',
        'Triffis C': '#bcbddc',
        'Schraubensalto A': '#057d59',
        'Schraubensalto': '#099e72',
        'Schraubensalto C': '#3fc49d',
        'Voll- ein 1 3/4 Salto vw C': '#637939',
        'Voll- ein- Rudi- aus B': '#8ca252',
        'Voll- ein- halb- aus B': '#b5cf6b',
        'Doppelsalto A': '#3182bd',
        'Doppelsalto B': '#6baed6',
        'Doppelsalto C': '#9ecae1',
        'Salto rw A': '#393b79',
        'Salto rw B': '#5254a3',
        'Salto rw C': '#6b6ecf',
        'Salto A': '#24b8bf',
        'Salto B': '#64ded2',
        'Salto C': '#a5e8e8',
        'Voll- ein- voll- aus A': '#31a354',
        'Voll- ein- voll- aus B': '#74c476',
        'Voll- ein- voll- aus C': '#a1d99b',
        '1/2 ein 1/2 aus C': '#636363',
        'Cody C': '#d9d9d9'}

    if shap_values is None:
        cmap = color_dict[shap_y_test]
    else:
        class_sequence = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
        d = dict(enumerate(np.array(shap_y_test.unique()).flatten(), 0))
        index_names = np.vectorize(lambda i: d[i])(class_sequence)

        list_cmap = np.vectorize(lambda i: color_dict[i])(index_names)
        cmap = ListedColormap(list_cmap)

    return cmap


def bar_plots(shap_values, shap_x_test, shap_y_test, bar='summary', size=None, jumps=None, folder=None, name=None, max_display=None):
    """
    :param shap_values: output from explainer.shap_values(shap_x_test)
    :param shap_x_test:
    :param shap_y_test:
    :param bar: 'summary', 'summary_color', 'single' or 'percentual', default: 'summary';
        different bar plots
    :param size: tuple (int, int);
        for individual plot size
    :param jumps: for 'single' and 'percentual' you can imput a list with the jump names, for that bar plots should be created
    :param folder: path, where the plots should be saved, whens its empty, plots will be shown and not be saved
    :param name: for saving the plot you can choose an other name
    :param max_display: int;
        for 'summary' and 'summary_color' plots, if its none, than all features will be display
    """

    if size is None:
        plot_size = {
            'summary': (25, 20),
            'summary_color': (25, 20),
            'single': (20, 20),
            'percentual': (30, 18)
        }
        size = plot_size[bar]
    if name is None:
        name = bar
    else:
        name = '-' + name
    if jumps is None:
        jumps = []
    if max_display is None:
        max_display = len(shap_x_test.columns)

    if bar == 'summary':
        if folder is None:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=ListedColormap('#616161'), class_names=shap_y_test.unique(),
                              max_display=max_display)
        else:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=ListedColormap('#616161'), class_names=shap_y_test.unique(),
                              max_display=max_display, show=False)
            plt.savefig(folder + name + '.png')
        plt.clf()
    elif bar == 'summary_color':
        if folder is None:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=create_colormap(shap_y_test, shap_values), class_names=shap_y_test.unique(),
                              max_display=max_display)
        else:
            shap.summary_plot(shap_values, shap_x_test, plot_type='bar', plot_size=size,
                              color=create_colormap(shap_y_test, shap_values), class_names=shap_y_test.unique(),
                              max_display=max_display, show=False)
            plt.savefig(folder + name + '.png')
        plt.clf()
    elif bar == 'single':
        if len(jumps) == 0:
            jumps = shap_y_test.unique()
        for jump in jumps:
            color_string = create_colormap(jump)
            shap.summary_plot(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__(), shap_x_test,
                              plot_type='bar', color=color_string, plot_size=size, show=False)
            if folder is None:
                plt.show()
            else:
                plt.savefig(folder + jump.replace('/', '-') + '.png')
            plt.clf()
    elif bar == 'percentual':
        feature_names = shap_x_test.columns
        jump_dict = {}
        if len(jumps) == 0:
            jumps = shap_y_test.unique().tolist()
            jumps.sort()
        for jump in jumps:
            values = np.abs(shap_values[np.where(shap_y_test.unique() == jump)[0][0]].__abs__()).mean(0)
            sum_values = np.sum(values)
            jump_dict[jump] = [(v / sum_values) * 100 for v in values]

        # creates the plot
        labels = list(jump_dict.keys())
        data = np.array(list(jump_dict.values()))
        data_cum = data.cumsum(axis=1)
        # rainbow or turbo cmap
        if len([item for item in feature_names if item.startswith('0')]) > 0:
            category_colors = plt.get_cmap('rainbow')(
                np.linspace(0.15, 0.85, len([item for item in feature_names if item.startswith('0')])))
            # how often category_colors have to repeat itself
            number_sub_cm = len([item for item in feature_names if not item.startswith('DJump')]) / \
                            len([item for item in feature_names if item.startswith('0')])
            category_colors = np.tile(category_colors, (int(number_sub_cm), 1))
        else:
            category_colors = np.ndarray(shape=(1, 4))
        count_djumps = len([item for item in feature_names if item.startswith('DJump')])
        # if Djumps in features, than other colormap must be appended
        if count_djumps > 0:

            dj_12_colors = np.array([
                [0.19216, 0.50980, 0.74118, 1.00000],
                [0.19216, 0.63922, 0.32941, 1.00000],
                [0.90196, 0.33333, 0.05098, 1.00000],
                [0.41961, 0.68235, 0.83922, 1.00000],
                [0.45490, 0.76863, 0.46275, 1.00000],
                [0.99216, 0.55294, 0.23529, 1.00000],
                [0.61961, 0.79216, 0.88235, 1.00000],
                [0.63137, 0.85098, 0.60784, 1.00000],
                [0.99216, 0.68235, 0.41961, 1.00000],
                [0.77647, 0.85882, 0.93725, 1.00000],
                [0.78039, 0.91373, 0.75294, 1.00000],
                [0.99216, 0.81569, 0.63529, 1.00000]])
            dj_9_colors = np.array([
                [0.48235, 0.25490, 0.45098, 1.00000],
                [0.64706, 0.31765, 0.58039, 1.00000],
                [0.80784, 0.42745, 0.74118, 1.00000],
                [0.45882, 0.41961, 0.69412, 1.00000],
                [0.61961, 0.60392, 0.78431, 1.00000],
                [0.73725, 0.74118, 0.86275, 1.00000],
                [0.88235, 0.07843, 0.50588, 1.00000],
                [0.96863, 0.50588, 0.74902, 1.00000],
                [0.96863, 0.73333, 0.82353, 1.00000]])

            # if nine djump pack not included
            if (count_djumps % 12) == 0:
                # if Djump features first, colormap must be appended on the first position
                if feature_names[0].startswith('DJump'):
                    category_colors = np.concatenate((
                        np.tile(dj_12_colors, (int(count_djumps/12), 1)),
                        category_colors))
                else:
                    category_colors = np.concatenate((
                        category_colors,
                        np.tile(dj_12_colors, (int(count_djumps/12), 1))))
            # if djump contains nine pack
            else:
                c = int((count_djumps - 9) / 12)
                sub_cm = np.concatenate((dj_9_colors, np.tile(dj_12_colors, (c, 1))))
                if feature_names[0].startswith('DJump'):
                    category_colors = np.concatenate((sub_cm, category_colors))
                else:
                    category_colors = np.concatenate((category_colors, sub_cm))

        fig, ax = plt.subplots(figsize=size)
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        for i, (colname, color) in enumerate(zip(feature_names.tolist(), category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.6,
                            label=colname, color=color)

            r, g, b, _ = color
            if len(feature_names) < 70:
                ax.legend(ncol=1, bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
            else:
                ax.legend(ncol=2, bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
        plt.suptitle('Percentual impact bar plot of the individual features')

        if folder is None:
            plt.show()
        else:
            plt.savefig(folder + name + '.png')
        plt.clf()

    def beeswarm(shap_values, shap_x_test, shap_y_test, jump, size=(25, 15), folder=None):
        """

        for beeswarm plots
        :param shap_values:
        :param shap_x_test:
        :param shap_y_test:
        :param jump:
        :param size: tuple (int,int);
            for individual size
        :param folder: default is None;
            if folder is None, plot will shown, else plot will be saved in the given path
        """

        jump_data = np.where(shap_y_test.unique() == jump)[0][0]
        shap.summary_plot(shap_values[jump_data], shap_x_test, plot_size=size, title=jump, show=False)
        if folder is None:
            plt.show()
        else:
            plt.savefig(folder + jump + '.png')

    def confusion_matrix(model, x_test, y_test, size=(35, 25), folder=None, name='confusion_matrix'):
        """
        for creating confusion matrix plot
        :param model: need the model
        :param x_test:
        :param y_test:
        :param size: (int,int);
            to adjust the size of the plot
        :param folder: default None or input path as a string;
            if folder is None, the plot will be shown and not saved,
            plot will be otherwise saved in this folder
        :param name: unique name for saving the plot
        """

        if len(y_test) == len(y_test.columns):
            # special colormap if there is for each class only one jump
            cmap_cm = ['#ffffff', '#048166']
            cmap_cm = ListedColormap(cmap_cm)
        else:
            cmap_cm = process_cmap('summer')
            cmap_cm.insert(0, '#ffffff')
            cmap_cm.insert(-1, '#000000')
            cmap_cm = ListedColormap(cmap_cm)
        cm = sklearn.metrics.confusion_matrix(y_test.idxmax(axis=1),
                                              pd.DataFrame(model.predict(x_test), columns=y_test.columns).idxmax(
                                                  axis=1))
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.columns)
        disp.plot(cmap=cmap_cm)
        disp.figure_.set_figwidth(size[0])
        disp.figure_.set_figheight(size[1])
        disp.figure_.autofmt_xdate()
        if folder is None:
            plt.show()
        else:
            plt.savefig(folder + name + '.png')
        plt.clf()

    def jump_core_plot(scores, percentage, min_y_value=70, size=(13, 13), title='', folder=None, name='jump_core_detection'):
        """
        creates a flying bar char for the jump core detection
        :param scores: dict with values
        :param percentage: int;
            percentage step size
        :param min_y_value: int between 0 and 70;
            remove y axis under min_y_value
        :param size: (int,int)
            to adjust plot size
        :param title: optional str;
            title on the plot
        :param folder: default None or input path as a string;
            if folder is None, the plot will be shown and not saved,
            plot will be otherwise saved in this folder
        :param name: unique name for saving the file
        :return:
        """
        full_list = [l for l in range(0, 100, percentage)]

        plt.figure(figsize=size)
        plt.suptitle(title)
        plt.xlabel('Data')
        plt.ylabel('Accuracy')
        plt.axis([0, full_list[-1], min_y_value, 100])
        plt.xticks(range(0, 100 + percentage, percentage))
        plt.yticks(range(min_y_value, 105, 5))
        plt.grid(True, axis='x')
        # cmap = process_cmap('brg', len(scores))

        for i in range(len(scores)):
            entry = list(scores.items())[i]
            start, end = entry[0].split('-')
            acc = entry[1] * 100
            if int(acc) >= min_y_value:
                if start.replace(' ', '') == '0':
                    plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#0000ff', alpha=0.7)
                elif end.replace(' ', '') == str(full_list[-1]):
                    plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#ff0000', alpha=0.7)
                else:
                    plt.axhline(acc, (int(start) / 100), (int(end) + percentage) / 100, color='#00ff00', alpha=0.7)
        if folder is None:
            plt.show()
        else:
            plt.savefig(folder + name + '.png')
        plt.clf()

