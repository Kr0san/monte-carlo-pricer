from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import os
import pyqtgraph as pg
from scipy.stats import gaussian_kde
from SimulationModels import *

path = os.path.abspath(os.getcwd()) + "\ic2.ico"
path2 = os.path.abspath(os.getcwd()) + "\math_ico.ico"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.statusBar()
        self.setGeometry(300, 100, 1200, 900)
        self.setMinimumSize(400, 400)
        self.setWindowTitle('Monte Carlo Simulator')
        self.setWindowIcon(QtGui.QIcon(path))
        
        self.toolbar()
        self.counter = 1

        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setFrameStyle(0)
        self.scrollArea.setWidgetResizable(True)

        self.tab_view = QtWidgets.QTabWidget()
        self.tab_view.setTabsClosable(True)
        self.tab_view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.tab_view.tabCloseRequested.connect(self.close_tab)
        self.scrollArea.setWidget(self.tab_view)
        self.setCentralWidget(self.scrollArea)

    def toolbar(self):
        tab_act = QtWidgets.QAction(QtGui.QIcon(path2), 'Create New Simulation Tab', self)
        tab_act.triggered.connect(self.add_tabs)
        tab_act.setStatusTip('New Simulation Tab')
        self.tool = self.addToolBar('New Pricing Window')
        self.tool.addAction(tab_act)

    def add_tabs(self):
        tab_name = "Simulation {}".format(self.counter)
        self.tab_view.addTab(InnerWidget(), tab_name)
        self.counter += 1

    def close_tab(self, index):
        self.tab_view.removeTab(index)


class InnerWidget(QtWidgets.QWidget):
    """ Inner widget class handles left and right panels. Left panel is composed of input boxes whereas right panel
        consists of simulation plot and histogram and density plot."""

    def __init__(self):
        super(InnerWidget, self).__init__()
        self.widget_interface()

    def widget_interface(self):

        # Diffusion group variable enables extra parameters for jump-diffusion model. These being average number of
        # jumps and average jump size. Widget itself is hidden by default unless jump-diffusion model is selected from
        # the combo box. Code segment below creates QGroupBox widget for diffusion model parameter.

        self.diffusion_group = QtWidgets.QGroupBox("Jump-Diffusion Parameters")
        lambda_label = QtWidgets.QLabel("Average number of jumps, <i>λ</i>:")
        self.lambda_edit = QtWidgets.QLineEdit()
        self.lambda_edit.setText('0')
        kappa_label = QtWidgets.QLabel("Average Size of Jumps, <i>κ</i>:")
        self.kappa_edit = QtWidgets.QLineEdit()
        self.kappa_edit.setText('0')
        delta_label = QtWidgets.QLabel("Volatility of jump process, <i>δ</i>:")
        self.delta_edit = QtWidgets.QLineEdit()
        self.delta_edit.setText('0')

        lambda_layout = QtWidgets.QHBoxLayout()
        lambda_layout.addWidget(lambda_label)
        lambda_layout.addWidget(self.lambda_edit)
        
        kappa_layout = QtWidgets.QHBoxLayout()
        kappa_layout.addWidget(kappa_label)
        kappa_layout.addWidget(self.kappa_edit)

        delta_layout = QtWidgets.QHBoxLayout()
        delta_layout.addWidget(delta_label)
        delta_layout.addWidget(self.delta_edit)
        
        diffusion_layout = QtWidgets.QVBoxLayout()
        diffusion_layout.addLayout(lambda_layout)
        diffusion_layout.addLayout(kappa_layout)
        diffusion_layout.addLayout(delta_layout)

        self.diffusion_group.setLayout(diffusion_layout)
        self.diffusion_group.setFixedHeight(120)
        self.diffusion_group.hide()

        # Variance-gamma group enables extra parameters for variance-gamma model. This group is hidden by default
        # unless variance-gamma model is selected. Code segment below creates QGroupBox widget for VG model parameters.

        self.variance_gamma_group = QtWidgets.QGroupBox("Variance-Gamma Parameters")
        nu_label = QtWidgets.QLabel("Variance Rate of Gamma Process, <i>ν</i>:")
        self.nu_edit = QtWidgets.QLineEdit()
        self.nu_edit.setText('0')
        theta_label = QtWidgets.QLabel("Skewness of Distribution, <i>θ</i>:")
        self.theta_edit = QtWidgets.QLineEdit()
        self.theta_edit.setText('0')

        nu_layout = QtWidgets.QHBoxLayout()
        nu_layout.addWidget(nu_label)
        nu_layout.addWidget(self.nu_edit)

        theta_layout = QtWidgets.QHBoxLayout()
        theta_layout.addWidget(theta_label)
        theta_layout.addWidget(self.theta_edit)

        variance_gamma_layout = QtWidgets.QVBoxLayout()
        variance_gamma_layout.addLayout(nu_layout)
        variance_gamma_layout.addLayout(theta_layout)

        self.variance_gamma_group.setLayout(variance_gamma_layout)
        self.variance_gamma_group.setFixedHeight(120)
        self.variance_gamma_group.hide()

        # Implementation of combo box for model selection

        self.combo_m = QtWidgets.QComboBox()
        self.combo_m.addItem("Geometric Brownian Motion")
        self.combo_m.addItem("Jump-Diffusion")
        self.combo_m.addItem("Variance-Gamma")

        self.combo_m.activated[str].connect(self.model_visibility)

        # Code below creates simulation model drop-down box and its layout

        self.model_group = QtWidgets.QGroupBox("Simulation Model")
        model_label = QtWidgets.QLabel("Model:")
        model_combo = self.combo_m
        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_combo)
        self.model_group.setLayout(model_layout)
        self.model_group.setFixedHeight(80)

        # Segment below creates simulation option parameters responsible for the settings of Monte Carlo simulation
        # and pricing of options.

        self.simulation_group = QtWidgets.QGroupBox("Simulation Options")
        steps_label = QtWidgets.QLabel("Time steps:")
        self.steps_edit = QtWidgets.QLineEdit()
        self.steps_edit.setText('365')
        self.steps_edit.setFixedWidth(146.5)
        simulations_label = QtWidgets.QLabel("Simulations:")
        self.simulations_edit = QtWidgets.QLineEdit()
        self.simulations_edit.setText('100000')
        self.simulations_edit.setFixedWidth(146.5)
        visible_plots_label = QtWidgets.QLabel("Visible Price Paths:")
        self.visible_plots_edit = QtWidgets.QLineEdit()
        self.visible_plots_edit.setText('10')
        self.visible_plots_edit.setFixedWidth(146.5)

        steps_layout = QtWidgets.QHBoxLayout()
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_edit)

        simulations_layout = QtWidgets.QHBoxLayout()
        simulations_layout.addWidget(simulations_label)
        simulations_layout.addWidget(self.simulations_edit)

        visible_plots_layout = QtWidgets.QHBoxLayout()
        visible_plots_layout.addWidget(visible_plots_label)
        visible_plots_layout.addWidget(self.visible_plots_edit)

        simulation_group_layout = QtWidgets.QVBoxLayout()
        simulation_group_layout.addLayout(steps_layout)
        simulation_group_layout.addLayout(simulations_layout)
        simulation_group_layout.addLayout(visible_plots_layout)

        self.simulation_group.setLayout(simulation_group_layout)

        # Implementation of base parameter group box. These are the main parameters that are used by all three models.
        # Risk-free interest rate and dividend yield controls drift (r-q).
        # Volatility controls rate of change of the underlying stochastic process
        # Start and end dates define time span of the simulation

        self.base_parameter_group = QtWidgets.QGroupBox("Base Parameters")
        price_label = QtWidgets.QLabel("Market Price, <i>S</i>:")
        self.price_edit = QtWidgets.QLineEdit()
        self.price_edit.setText('100')
        self.price_edit.setFixedWidth(146.5)
        rate_label = QtWidgets.QLabel("Risk-Free Rate, <i>r</i>:")
        self.rate_edit = QtWidgets.QLineEdit()
        self.rate_edit.setText("0")
        self.rate_edit.setFixedWidth(146.5)
        volatility_label = QtWidgets.QLabel("Volatility, <i>σ</i>:")
        self.volatility_edit = QtWidgets.QLineEdit()
        self.volatility_edit.setText('0')
        self.volatility_edit.setFixedWidth(146.5)
        dividend_label = QtWidgets.QLabel("Dividend Yield Rate, <i>q</i>:")
        self.dividend_edit = QtWidgets.QLineEdit()
        self.dividend_edit.setText('0')
        self.dividend_edit.setFixedWidth(146.5)
        start_date_label = QtWidgets.QLabel("Start Date:")
        self.start_date_edit = QtWidgets.QDateTimeEdit(QtCore.QDate(2000, 1, 1))
        self.start_date_edit.setDisplayFormat('yyyy-MM-dd')
        end_date_label = QtWidgets.QLabel("Expiry Date:")
        self.end_date_edit = QtWidgets.QDateTimeEdit(QtCore.QDate.currentDate())
        self.end_date_edit.setDisplayFormat('yyyy-MM-dd')

        price_layout = QtWidgets.QHBoxLayout()
        price_layout.addWidget(price_label)
        price_layout.addWidget(self.price_edit)

        rate_layout = QtWidgets.QHBoxLayout()
        rate_layout.addWidget(rate_label)
        rate_layout.addWidget(self.rate_edit)

        volatility_layout = QtWidgets.QHBoxLayout()
        volatility_layout.addWidget(volatility_label)
        volatility_layout.addWidget(self.volatility_edit)

        dividend_layout = QtWidgets.QHBoxLayout()
        dividend_layout.addWidget(dividend_label)
        dividend_layout.addWidget(self.dividend_edit)

        start_date_layout = QtWidgets.QHBoxLayout()
        start_date_layout.addWidget(start_date_label)
        start_date_layout.addWidget(self.start_date_edit)

        end_date_layout = QtWidgets.QHBoxLayout()
        end_date_layout.addWidget(end_date_label)
        end_date_layout.addWidget(self.end_date_edit)

        base_parameter_group_layout = QtWidgets.QVBoxLayout()
        base_parameter_group_layout.addLayout(price_layout)
        base_parameter_group_layout.addLayout(rate_layout)
        base_parameter_group_layout.addLayout(volatility_layout)
        base_parameter_group_layout.addLayout(dividend_layout)
        base_parameter_group_layout.addLayout(start_date_layout)
        base_parameter_group_layout.addLayout(end_date_layout)

        self.base_parameter_group.setLayout(base_parameter_group_layout)
        self.base_parameter_group.setFixedHeight(210)

        # This adds option payoff selection combo box with Call and Put payoff
        # Output box updates with the price of option after Monte Carlo simulation is ran

        self.combo_opt = QtWidgets.QComboBox()
        self.combo_opt.addItem("Call")
        self.combo_opt.addItem("Put")

        self.option_group = QtWidgets.QGroupBox("Option pricing")
        option_label = QtWidgets.QLabel("Payoff Type:")
        option_combo = self.combo_opt
        option_layout = QtWidgets.QHBoxLayout()
        option_layout.addWidget(option_label)
        option_layout.addWidget(option_combo)

        opt_price_label = QtWidgets.QLabel("Option Price:")
        self.opt_price_edit = QtWidgets.QLineEdit()
        self.opt_price_edit.setText('0')
        self.opt_price_edit.setReadOnly(True)
        self.opt_price_edit.setFixedWidth(146.5)

        exercise_label = QtWidgets.QLabel("Exercise Price:")
        self.exercise_edit = QtWidgets.QLineEdit()
        self.exercise_edit.setText('0')
        self.exercise_edit.setFixedWidth(146.5)

        opt_price_layout = QtWidgets.QHBoxLayout()
        opt_price_layout.addWidget(opt_price_label)
        opt_price_layout.addWidget(self.opt_price_edit)

        exercise_layout = QtWidgets.QHBoxLayout()
        exercise_layout.addWidget(exercise_label)
        exercise_layout.addWidget(self.exercise_edit)

        option_group_layout = QtWidgets.QVBoxLayout()
        option_group_layout.addLayout(option_layout)
        option_group_layout.addLayout(exercise_layout)
        option_group_layout.addLayout(opt_price_layout)

        self.option_group.setLayout(option_group_layout)

        # This triggers simulation and pricing

        simulate_button = QtWidgets.QPushButton("Simulate", self)
        simulate_button.setStatusTip("Run Simulation")
        simulate_button.clicked.connect(self.simulation)

        # Code segment below creates main layout which is composed of model selection and parameters panel on the left
        # side and simulation/distribution plots on the right side. Splitter is added for extra interface functionality
        # and visual design

        left_panel = QtWidgets.QWidget()
        left_panel_layout = QtWidgets.QVBoxLayout()
        left_panel_layout.addWidget(self.model_group)
        left_panel_layout.addWidget(self.simulation_group)
        left_panel_layout.addWidget(self.base_parameter_group)
        left_panel_layout.addWidget(self.diffusion_group)
        left_panel_layout.addWidget(self.variance_gamma_group)
        left_panel_layout.addWidget(self.option_group)
        left_panel_layout.addWidget(simulate_button)
        left_panel_layout.setAlignment(QtCore.Qt.AlignTop)
        left_panel.setLayout(left_panel_layout)
        left_panel.setFixedWidth(350)

        # This designs left panel plots
        pg.setConfigOption('background', 'w')
        self.sim_plot = SimulationPlot()

        right_panel = QtWidgets.QWidget()
        right_panel_layout = QtWidgets.QVBoxLayout()
        right_panel_layout.addWidget(self.sim_plot)
        right_panel.setLayout(right_panel_layout)

        main_layout = QtWidgets.QHBoxLayout()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def model_visibility(self, selection):
        """ This function controls visibility of jump-diffusion and variance gamma layouts."""

        if selection == "Jump-Diffusion":
            self.diffusion_group.setVisible(True)
            self.variance_gamma_group.setVisible(False)
        elif selection == "Variance-Gamma":
            self.variance_gamma_group.setVisible(True)
            self.diffusion_group.setVisible(False)
        else:
            self.diffusion_group.setVisible(False)
            self.variance_gamma_group.setVisible(False)

    def simulation(self):
        """ This function is triggered whenever simulate button is pressed. It performs plotting of simulation and 
            histogram plots. Inputs are validated first and error handling is imposed in the form of warning boxes."""

        try:
            # Initialize simulation and plot parameters
            init_steps = int(self.steps_edit.text())
            init_sim = int(self.simulations_edit.text())
            init_plots = int(self.visible_plots_edit.text())

            # Initialize price, rate, dividend yield and volatility variables in the form of floats
            init_price = float(self.price_edit.text())
            init_rate = float(self.rate_edit.text())
            init_volatility = float(self.volatility_edit.text())
            init_div = float(self.dividend_edit.text())

            # Initialize strike price
            init_exercise = float(self.exercise_edit.text())

            # Validate inputs
            if init_volatility <= 0 or init_price <= 0:
                raise ValueError

            if init_steps <= 0 or init_sim <= 0:
                raise ValueError

            # Validate dates. Start date cannot exceed end date
            date1 = dt.datetime.strptime(self.start_date_edit.text(), '%Y-%m-%d')
            date2 = dt.datetime.strptime(self.end_date_edit.text(), '%Y-%m-%d')

            if date2 - date1 <= dt.timedelta(0):
                raise RuntimeError

            simulation_model = SimulationModels(init_price, self.start_date_edit.text(), self.end_date_edit.text(),
                                                init_rate, init_volatility, init_div, init_sim, init_steps)
            self.sim_plot.simulation_plot.clear()

            if self.combo_m.currentText() == "Geometric Brownian Motion":
                data = simulation_model.geometric_brownian_motion()

            elif self.combo_m.currentText() == "Jump-Diffusion":
                # Initialize jump diffusion parameters
                init_lambda = float(self.lambda_edit.text())
                init_kappa = float(self.kappa_edit.text())
                init_delta = float(self.delta_edit.text())

                data = simulation_model.jump_diffusion(init_lambda, init_kappa, init_delta)

            else:
                # Initialize variance gamma parameters
                init_theta = float(self.theta_edit.text())
                init_nu = float(self.nu_edit.text())

                if init_nu == 0.0:
                    raise ValueError

                data = simulation_model.variance_gamma(init_theta, init_nu)

            # This calculates option price
            opt = self.calculate_option_price(simulation_model.european_option, data, init_exercise,
                                              self.combo_opt.currentText())
            self.opt_price_edit.setText(str(opt))

            # This plots simulation and density plot
            for i, k in zip(data[0:init_plots, :], range(len(data))):
                self.sim_plot.simulation_plot.plot(i, pen=k, antialias=True)

            self.sim_plot.plot_density(data, 50)

        except ValueError:
            self.error_box1()

        except RuntimeError:
            self.error_box2()

    def error_box1(self):
        QtWidgets.QMessageBox.information(self, "Error", "Invalid model parameters.")

    def error_box2(self):
        QtWidgets.QMessageBox.information(self, "Error", "End date should be greater than the start date.")

    @staticmethod
    def calculate_option_price(option_type, model, exercise, payoff_type):
        return option_type(model, exercise, payoff_type)


class SimulationPlot(pg.GraphicsView):
    """ This class subclasses GraphicView from Pyqtgraph and constructs two plots. One for simulation and another for
        histogram/density plot. It is located in the right panel."""

    def __init__(self):
        super(SimulationPlot, self).__init__()
        self.setMinimumSize(600, 400)
        self.plot_layout = pg.GraphicsLayout(border=(100, 100, 100))
        self.setCentralItem(self.plot_layout)
        title = "Simulation and distribution plots"
        self.plot_layout.addLabel(title, col=1)
        self.plot_layout.setContentsMargins(5, 5, 5, 5)
        self.plot_layout.nextRow()

        self.simulation_plot = self.plot_layout.addPlot(title="Simulated Price Paths", col=1)
        self.simulation_plot.showGrid(x=True, y=True)
        self.simulation_plot.setLabel('left', 'Price')
        self.simulation_plot.setLabel('bottom', 'Time steps')
        self.plot_layout.nextRow()

        self.density_plot = self.plot_layout.addPlot(title="Distribution of Simulated Prices", col=1)
        self.density_plot.showGrid(x=True, y=True)
        self.density_plot.setLabel('left', 'Probability')
        self.density_plot.setLabel('bottom', 'Price')

    def plot_density(self, data, factor):
        """ This function plots histogram and associated pdf. Each call clears previous plot field."""

        self.density_plot.clear()

        values = np.hstack(data)
        hist, bins = np.histogram(values, bins='auto', density=True)
        self.density_plot.plot(bins, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150), antialias=True)

        kde = gaussian_kde(values, bw_method='silverman')
        dist_space = np.linspace(min(values), max(values), factor)

        self.density_plot.plot(dist_space, kde(dist_space), pen=pg.mkPen('r', width=2), antialias=True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
