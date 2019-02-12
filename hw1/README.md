## Artificial Intelligence Homework 1

Spencer Hann

CS 541 | Winter 2019

Portland State University



### 1.

Assuming $A​$ is invertible,
$$
\begin{align}
(A^{-1} A) &= I \\
(A^{-1} A)^T &= I^T \\
(A^{-1})^T A^T &= I \\
(A^{-1})^T A^T (A^T)^{-1} &= I (A^T)^{-1} \\
(A^{-1})^T &= (A^T)^{-1}
\end{align}
$$

------------------------------------------------------

### 2.

$$
\begin{align}
\text {Bayes' Rule :} \quad
P(H|e) &= \frac {P(e|H)\ P(H)} {P(e)} \\
P(disease\ |\ test\ positive) &= \frac{P(positive\ |\ disease) \cdot P(disease)}
{P(positive)} \\
P(disease\ |\ positive) &= \frac{P(positive\ |\ disease) \cdot P(disease)}
{P(true\ positive)\ +\ P(false\ positive)} \\
P(disease\ |\ positive) &= \frac{99\% \cdot \frac 1 {10,000}} 
{99\% \cdot 1\ +\ 1\% \cdot 9,999} \\
P(disease\ |\ positive) &= \frac{0.000099} 
{0.99 +\ 100} \\
P(disease\ |\ positive) &= 0.00000098
\end{align}
$$

-------------------------------------------------------

### 3.

$$
f(x\ |\ \mu,\ \sigma^2) = \frac 1 {\sqrt{2 \pi \sigma^2}} ^* \cdot e^{-\frac {(x-\mu)^2} {2 \sigma^2}} \\
\begin{align}
f'(x\ |\ \mu,\ \sigma) &= \frac d {dx} e^{-\frac {(x-\mu)^2} {2 \sigma^2}} \\
&= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot 
\frac d {dx} \Bigg[-\frac {(x-\mu)^2} {2 \sigma^2} \Bigg] \\
&= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot \frac d {dx} \Bigg[(x-\mu)^2 \Bigg] \cdot -\frac 1 {2 \sigma^2} ^* \\
&= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot \frac d {dx} (x-\mu)^2 \\
&= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot 2(x-\mu) \cdot \frac d {dx} (x-\mu) \\
&= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot 2(x-\mu) \cdot (1-0) \\
&= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot 2(x-\mu) \\
So, \\
0 &= e^{-\frac {(x-\mu)^2} {2 \sigma^2}}\ \cdot 2(x-\mu) \\
0 &= x-\mu \\
x &= \mu \\

\end{align} \\

\\ ^*
\text {  The }
\frac 1 {\sqrt{2 \pi \sigma^2}\ } \text { and }\ \frac 1 {2 \sigma^2} \text{ terms can be ignored because they } \\
\text {are constants, given the distribution. They will affect the } \\
\text {maximum, but not the $x$ where the maximum occurs} \\
$$

------------------------------------------------------

### 4.

#### (i)

$$
\begin{align}
E[X] &= \sum_i x_i f(x_i) \\
E[X] &= \sum_{i=1}^4 x_i f(x_i) \\
E[X] &= 4 (1 \cdot 0.5) \\
E[X] &= 2
\end{align}
$$

#### (ii)

Let $x_i = i$, so that $x_i$ the $i$th coin flip.

Let $f(x_i) = \big(\frac 1 {2}\big)^i$, so that $f(x_i)$ is the probability of $i$ tails in a row (no heads yet).
$$
\begin{align}
E[X] &= \sum_i x_i f(x_i) \\
&= \sum_{i=1}^\infty\ i\, \bigg(\frac 1 {2}\bigg)^i \\
&= x_1 \bigg(\frac {1 - (\frac 1 2)^\infty} {1 - \frac 1 2}\bigg) \\
&= 1 \bigg(\frac {1 - 0} {1 - \frac 1 2}\bigg) \\
&= \frac {1} {1 - \frac 1 2} \\
&= \frac {1} {\frac 1 2} \\
&= 2
\end{align}
$$

-----------------------------------------

### 5.  _(2.2)_

#### a.

We can see that the agent is acting rationally with respect to its percept sequence. Whenever it perceives dirt it performs the _Suck_ action, regardless of its location. When it perceives a clean square, it moves either _Right_ or _Left_ if it is in squares _A_ or _B_, respectively, meaning it moves to the other square if there is no dirt to be cleaned.

Put simply, knowing whether or not to _Suck_ and knowing to switch squares when the current square is clean are the only requirements to maximizing this performance measure.

#### b.

Yes. The agent would need to store all previously cleaned squares, so that it did not revisit a square. The initial version of the agent would continuously move back and forth checking for dirt, losing all its points.

#### c.

It would be useful for the agent to be able to learn/map the geography of the environment. How this information affects its _percept sequence_ $\rightarrow$ _action_ function depends on the nature with which squares become dirty again. It it is totally random, then there is nothing for the agent to learn, other than how to not hit a wall and possibly a best path through the environment hitting all or most squares. However, if there is a pattern to how the squares become dirty again, it would be useful for the agent to be able to learn this pattern and adjust its behavior.

------------------------------------------------------

### 6.  _(2.3)_

#### a.

False. Not all information is necessary to act rationally. The agent only needs _relevant_ information about the environment, where relevance is defined with respect to its performance measure. Even if an agent does not have access to all relevant information about the environment at all times, it may be able to make up for this by storing information about the environment internally.

### b.

True. Some task environments are complex enough that an agent needs to store a state history, rather than simply reacting to the current world state, like a card matching game.

### c.

True. It depends how "rational" is defined for the task environment. If all action give a reward, then all agents are rational.

### d.

False. The agent program takes the current precept as input, whereas the agent function takes the entire percept history.

### e.

False. An agent function, which is a mathematical object, may be to large or complex to be implemented exactly with a physical machine, i.e. a function that maps all Go board states to the best move in that state. The physical agent program/machine implementation must use some heuristics to determine best moves.

### f.

True. Because the environment is deterministic, the agent should be able to know/predict future states and perform rational actions rather than acting randomly. However, at least one environment exists in which this agent is rational and that is an environment that always gives rewards.

### g.

True. It is possible that an agent designed for one task environment might still act in a way that is useful in another. Even if only, seemingly, by accident.

### h.

False. Some tasks are still achievable in an unobservable environment, so some agents can be rational, but not every agent will behave rationally with out input from the environment.

### i.

False, but this depends on the definition of "loses". If it means "never loses any money", the agent function would be to fold immediately, every have. If "never loses" means "wins every hand", then the even a perfectly rational agent cannot always avoid losing, because the environment is stochastic.

------------------------------------------------

### 7. _(2.4)_

2.4 For each of the following activities, give a PEAS description of the task environment and characterize it in terms of the properties listed in Section 2.3.2.

#### Playing soccer.

​	Fully observable; multi-agent; deterministic; sequential; dynamic; continuous; known

​	Performance Measure:

​		Score, making goals, preventing opposing team goals

​	Environment:

​		Ball, teammates, opponents, out of bound lines, goal

​	Actuators:

​		legs, arms (if goalie)

​	Sensors:

​		Camera

##### Exploring the subsurface oceans of Titan.

​	Partially observable; single agent; stochastic; sequential; dynamic; continuous; known

​	Performance Measure:

​		Total area mapped

​	Environment:

​		Water, ice, rock

​	Actuators:

​		Propeller

​	Sensors:

​		Camera, radar, lidar, sonar, thermometer, pressure gauge

##### Shopping for used AI books on the Internet.

​	Partially observable; single agent; deterministic; episodic; static; discrete; known

​	Performance Measure:

​		Money saved, book accuracy/quality (edition number/new/used)

​	Environment:

​		Internet

​	Actuators:

​		Display

​	Sensors:

​		Keyboard

##### Playing a tennis match.

​	Fully observable; multi-agent; deterministic; sequential; dynamic; continuous; known

​	Performance Measure:

​		Points

​	Environment:

​		Out of bounds lines, net, opponent, ball

​	Actuators:

​		tennis racket, jointed arm, wheels/legs

​	Sensors:

​		camera

##### Practicing tennis against a wall.
​	Fully observable; single agent; deterministic; sequential; dynamic; continuous; known

​	Performance Measure:

​		Points (number of legal bounces against wall?)

​	Environment:

​		out of bounds line, ball, wall, "net" (line on wall?)

​	Actuators:

​		tennis racket, jointed arm, wheels/legs

​	Sensors:

​		camera

##### Performing a high jump.

​	Fully observable; single agent; deterministic; sequential; static; continuous; known

​	Performance Measure:

​		height of bar, successfully clearing bar

​	Environment:

​		track, sand, crash pad bar

​	Actuators:

​		legs

​	Sensors:

​		camera, orientation sensor

##### Knitting a sweater.
​	Fully observable; single agent; deterministic; sequential; static; discrete; known

​	Performance Measure:

​		coziness

​	Environment:

​		knitting pattern, yarn, needles, armchair, re-runs of old TV shows

​	Actuators:

​		arms/hands, needles

​	Sensors:

​		camera

##### Bidding on an item at an auction.

​	Fully observable; multi-agent; deterministic; sequential; static; discrete; known

​	Performance Measure:

​		minimizing cost, successfully obtaining item

​	Environment:

​		auctioneer, opponent bidders

​	Actuators:

​		signal of bid

​	Sensors:

​		input feed of other bidders, current best bid

--------------------------------------------------------

### 8. _(2.6)_

#### a. 

Yes. While implementing an agent function there are many different design decisions to be made. For example, one system, in the interest of preserving memory, may re-compute the action response to a percept sequence, for every new percept/time frame. In contrast, another system may, in the interest of minimizing runtime, might store all agent function inputs and outputs in a massive lookup table.

#### b.

Yes. There are agent functions that are to complex to be implemented with physical systems. For example, an agent function that outputs future positions of every atom in the universe.

####  c. 

An function by definition maps all inputs to exactly one output, the the outputs are not necessarily unique to that input. If an agent program was implementing more that one agent function, then some input/percept might not have exactly one output/action. This is not possible by definition. Each agent program can implement only one agent function.

#### d. 

Given the set of all actions, $A$, there are $|A|^{2^{n}}$ possible agent programs.

#### e. 

No, but it could change the agent program's behavior/outputs. For example, thought the underlying function is the same, it may make different/better decisions because it has improved reaction time and can make more decisions per second.



-----------------------------

### 9. _(2.7)_

Write pseudocode agent programs for the goal-based and utility-based agents.
The following exercises all concern the implementation of environments and agents for the vacuum-cleaner world.



-------------------------------

### 10.













