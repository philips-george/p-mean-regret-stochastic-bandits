# p-mean-regret-stochastic-bandits
Code for running experiments described in the extended version of the paper "p-Mean Regret for Stochastic Bandits" by Anand Krishna, Philips George John, Adarsh Barik, Vincent Y.F. Tan.

These experiments can be run using the following command:
```
python pmean_expts.py [-p P_PARAM] [-T ROUNDS] [-n NUM_REPS] [-k NUM_ARMS] {bernoulli,beta,triangular,uniform}
```

The arguments are as follows:

```-p``` specifies the $p$ parameter in $p$-mean regret. We use $p \in \\{1,0.5,0,-0.5,-1,-2\\}$ in our experiments.

```-T``` specifies the number of rounds for the bandit algorithms. We use $T = 20,000$ and $T = 100,000$ (only for Bernoulli instance) in our experiments.

```-n``` specifies the number of repetitions to use in computing the inner-expectation in the $p$-mean regret definition. We use $30$ repetitions in all our experiments.

```-k``` specifies the number of arms in the bandit instance. We use $k = 50$ throughout.

The only required argument is the instance type, which can be ```bernoulli```, ```triangular```, ```beta``` or ```uniform```. These random synthetic stochastic bandit instances are defined in the paper itself.
