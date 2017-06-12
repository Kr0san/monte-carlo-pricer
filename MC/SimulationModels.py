import numpy as np
import datetime as dt


class SimulationModels:

    def __init__(self, price, start, end, r, sigma, div, sim, steps):
        self.price = price
        self.start = start
        self.end = end
        self.r = r
        self.sigma = sigma
        self.div = div
        self.sim = sim
        self.steps = steps
        self.T = (dt.datetime.strptime(self.end, '%Y-%m-%d') - dt.datetime.strptime(self.start, '%Y-%m-%d')).days / 365
        self.dt = self.T / self.steps
        self.discount_factor = np.exp(-self.r * self.T)

    def geometric_brownian_motion(self):

        c1 = (self.r - self.div - 0.5 * self.sigma ** 2) * self.dt
        c2 = self.sigma * np.sqrt(self.dt)
        bm = np.random.standard_normal(size=(self.sim, self.steps))

        s_ = np.zeros((self.sim, self.steps + 1))
        s_[:, 0] = self.price
        s_[:, 1:] = self.price * np.exp(np.cumsum(c1 + c2 * bm, axis=1))

        return s_

    def jump_diffusion(self, lamb, kappa, delta):

        rj = lamb * (np.exp(kappa + 0.5 * delta ** 2) - 1)
        c1 = (self.r - self.div- rj - 0.5 * self.sigma ** 2) * self.dt
        c2 = self.sigma * np.sqrt(self.dt)

        s_ = np.zeros((self.sim, self.steps + 1))
        bm1 = np.random.standard_normal(size=(self.sim, self.steps + 1))
        bm2 = np.random.standard_normal(size=(self.sim, self.steps + 1))
        poisson = np.random.poisson(lamb * self.dt, size=(self.sim, self.steps + 1))

        s_[:, 0] = self.price

        for i in range(1, self.steps + 1):
            s_[:, i] = s_[:, i - 1] * (np.exp(c1 + c2 * bm1[:, i]) +
                (np.exp(kappa + delta * bm2[:, i]) - 1) * poisson[:, i])

        return s_

    def variance_gamma(self, theta, nu):

        omega = (1/nu)*np.log(1 - theta*nu - (self.sigma**2 * nu)/2)
        s_ = np.zeros((self.sim, self.steps + 1))
        bm = np.random.standard_normal(size=(self.sim, self.steps))
        gamma = np.random.gamma(self.dt/nu, nu, size=(self.sim, self.steps))
        comb = theta*gamma + self.sigma*np.sqrt(gamma)*bm

        s_[:, 0] = self.price
        s_[:, 1:] = self.price * np.exp(np.cumsum((self.r - self.div + omega) * self.dt + comb, axis=1))

        return s_

    def european_option(self, model, strike, opt_type="Call"):

        if opt_type == "Call":
            payoffs = np.maximum(model[:,-1] - strike, 0)
        else:
            payoffs = np.maximum(strike - model[:, -1], 0)

        option_price = round(np.mean(payoffs) * self.discount_factor, 2)

        return option_price


if __name__ == "__main__":

    test = SimulationModels(100, '2015-04-30', '2017-04-30', 0.05, 0.25, 100000, 365)
    #print(test.geometric_brownian_motion())
    #print(test.gen_sn(3, 5))
    print(test.european_option(test.geometric_brownian_motion(), 110, opt_type="Call"))
    print(test.european_option(test.geometric_brownian_motion(), 110, opt_type="Put"))
    # plt.plot(test.jump_diffusion(1, 0.005, 0.3)[0,:])
    # plt.plot(test.jump_diffusion(1, 0.5, 0.3)[1,:])
    # plt.plot(test.variance_gamma(0.34, 0.0012)[1, :])
    # plt.show()
    # print(test.european_option(test.geometric_brownian_motion(), 120))
    # print(test.european_option(test.variance_gamma(0.3, 0.001), 120))
    # print(test.european_option(test.jump_diffusion(0, 0, 0), 120))
