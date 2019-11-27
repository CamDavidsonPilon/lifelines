# -*- coding: utf-8 -*-
f, axes = plt.subplots(2, 1)
ax = axes[0]

H = lambda t, lambda_, rho: (t / lambda_) ** rho
S = lambda t, lambda_, rho: np.exp(-H(t, lambda_, rho))

t = np.linspace(0.001, 10, 200)
ax.set_title(r"Weibull survival functions upon varying the parameters")
ax.plot(t, S(t, 0.5, 1), label=r"$\lambda=0.5, \rho=1$")
ax.plot(t, S(t, 1, 1), label=r"$\lambda=1, \rho=1$")
ax.plot(t, S(t, 2, 1), label=r"$\lambda=2, \rho=1$")
ax.legend()
ax.set_ylabel(r"$S(t)$")

ax = axes[1]

ax.plot(t, S(t, 1, 0.5), label=r"$\lambda=1, \rho=0.5$")
ax.plot(t, S(t, 1, 1), label=r"$\lambda=1, \rho=1$")
ax.plot(t, S(t, 1, 2), label=r"$\lambda=1, \rho=2$")
ax.legend()
ax.set_ylabel(r"$S(t)$")
ax.set_xlabel(r"$t$")

plt.savefig("weibull_parameters.png")
