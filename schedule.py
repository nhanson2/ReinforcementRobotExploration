import numpy as np
class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: initial value
        :param value_to: final value
        :param num_steps: number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps
        self.a = self.value_from
        self.b = np.log(value_to/self.a)/(num_steps-1)
        
    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step:  The step at which to compute the interpolation.
        :rtype: float.  The interpolated value.
        """

        # YOUR CODE HERE:  implement the schedule rule as described in the docstring,
        # using attributes `self.a` and `self.b`.
        if step <= 0:
            value = self.value_from
        elif step >= self.num_steps-1:
            value = self.value_to
        else:
            value = self.a * np.exp(self.b*step)
        
        return value
    
class OUSchedule:
    def __init__(self, steps, action_space_dim, mu=0.0, standard_deviation=0.2, dt=0.01):
        """Ornstein-Uhlenbeck based exploration for continuous process

        :param mu: mean value of noise
        :param action_space_dim: dimension of the action space
        :param standard_deviation: standard deviation of noise
        :param dt: step size to take
        """
        self.steps = steps
        self.action_space_dim = action_space_dim
        self.mu = mu
        self.std = standard_deviation
        self.weights = np.zeros((steps+1, self.action_space_dim))
        self.dw = np.zeros((steps+1, self.action_space_dim))
        self.dw[0,:] = np.random.normal(self.mu,self.std,self.action_space_dim)
        self.weights[0,:] = self.dw[0,:]
        self.dt = dt
        
    def value(self, step) -> float:
        """Calculates the next step in a random walk Gaussian process (equivalent to maintaining momementum)

        :param step:  The step at which to compute the process
        :rtype: float.  The process noise step
        """
        self.dw[step+1] = np.sqrt(self.dt)*np.random.normal(self.mu,self.std,self.action_space_dim)
        self.weights[step+1,:] = self.weights[step,:] + self.dw[step+1,:]
        return self.weights[step+1,:]
            
    def reset(self) -> None:
        """Reset the process to the start state
        """
        self.weights = np.zeros((self.steps+1, self.action_space_dim))
        self.dw = np.zeros((self.steps+1, self.action_space_dim))
        self.dw[0,:] = np.random.normal(self.mu,self.std)
        self.weights[0,:] = self.dw[0,:]