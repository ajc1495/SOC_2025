# SOC_2025
# WEEK - 1 : Introduction to DRL & Financial Markets
# David Silver’s RL Lecture
1. Agent–Environment Loop: 
An agent observes a state, takes an action, receives a reward, and transitions to a new state. Over time, it learns which actions maximize cumulative reward. 

2. Key Terminology & Concepts:
I have explored notions like policy (mapping states to actions), value functions (estimating long-term reward), and the trade-off between exploring new actions versus exploiting known strategies.

3. Foundational Mathematics: 
I saw how Reinforcement Learning is formalized using Markov Decision Processes (MDPs), which provide a mathematical framework for decision-making under uncertainty.

# Spinning Up in Deep RL
1. What Is RL?
Reinforcement Learning is about teaching agents to make sequential decisions through trial-and-error interactions with the environment. 

2. Key Concepts Covered:
Clear definitions for agent, environment, state, action, reward, policy, value function, and return.
*Formal structure using MDPs and definitions of stochastic vs deterministic policies. 

3. Taxonomy of RL Algorithms:
Spinning Up outlines the major families: value-based (e.g., Q‑learning), policy-based (e.g., policy gradients), and actor–critic methods.

5. Policy Optimization Primer:
Introduces policy gradient techniques: deriving and implementing the simplest policy gradient, understanding the log‑prob trick, and preventing gradient issues (e.g., by ignoring irrelevant past trajectories). 

5. Hands‑On Support:
Includes code templates, exercises, and references to key algorithms (TRPO, PPO, SAC, etc.) ideal for researchers getting practical experience.

# Modern Portfolio Theory

1. Core Definition:
A financial framework enabling risk-averse investors to construct a portfolio that maximizes expected return for a given level of risk by balancing assets. 

2. Key Principles:
Diversification is central—it reduces risk by combining assets with low or negative correlation, lowering overall portfolio volatility. 
Returns are based on weighted averages; portfolio risk depends on individual variances and correlations among assets. 

3. Efficient Frontier & CAPM:
The efficient frontier represents optimal portfolios offering the best return for each risk level. 
Introducing a risk-free asset like government bonds creates the Capital Allocation Line (CAL), identifying optimal combinations. 
This theory underpins the Capital Asset Pricing Model (CAPM), which links expected returns to risk (beta).

4. Advantages & Limitations:
MPT promotes creating well-diversified portfolios using tools like ETFs and target-date funds. 

5. Common criticisms: Reliance on variance as a risk measure (not distinguishing upside vs downside), assumptions of normal distribution, and accurate historical estimates. → Post‑Modern Portfolio Theory (PMPT) addresses these issues.

# WEEK - 2 : Data Collection & Processing
# yfinance Python Tutorial

yfinance is a widely used, open-source Python library for fetching financial data from Yahoo Finance . Key takeaways are as follows:

1. Installation & Setup:   
* Install easily via pip install yfinance or from conda-forge, and it returns data in pandas.DataFrame format. 

2. Core Classes & Methods:
* Use yf.Ticker(ticker_symbol) to fetch detailed data.
* Access methods like .history() for price data, .actions, .dividends, .splits, and .info for company metadata.
* .download() lets you efficiently retrieve data for multiple tickers at once, with options for interval, threading, and auto-adjustment.

3. Data Types Available:
* Historical prices, dividends, splits, fundamentals, analyst ratings, institutional holders, and options chains using .options and .option_chain().

4. Strengths & Caveats:
* It’s ideal for research or backtesting, but not recommended for live trading due to potential delays, rate limitations, and reliance on unofficial scraping.

# Pandas Time Series & Date Functionality

The pandas documentation (v2.3) covers extensive time-series support using datetime64 and timedelta64 types. Highlights include:

1. Parsing and Date‑time Conversion: 
* Convert strings, NumPy, or Python datetime objects into DatetimeIndex using pd.to_datetime().

2. Date‑time Indexing & Slicing:
* Set a DatetimeIndex as the DataFrame index for simple slicing (e.g., df['2023-06':'2023-07']).

3. Accessing Date Components:
* Use .dt accessor to extract attributes like .year, .month, .weekday, enabling group-by operations, filtering, and feature engineering.

4. Time Zone Management:
* Use .tz_localize() and .tz_convert() to work with time zone–aware series.

5. Resampling & Frequency Conversion: 
* Change data frequency via resample(), applying aggregations like .mean(), .max(), and .sum().

6. Plotting Time Series:
* Integrated plotting with pandas (e.g. line plots) works seamlessly on time-indexed data.

7. Summary:
* Overall, pandas provides robust tools for handling time-stamped data: parsing, indexing, extracting components, timezone conversion, resampling, grouping, and visualization.

# WEEK - 3 : Technical Indicators + Custom Gym Environment

# TA‑Lib Python
Provides a Python wrapper around TA‑Lib, a powerful C/C++ library with over 150 built-in technical indicators and candlestick pattern recognition.
1. Function API & Abstract API: Use simple functions like RSI, MACD, or BBANDS() that operate on Pandas or NumPy arrays.
2. Installation: Needs the underlying C library; installable via pip or conda.
3. Speed: Cython wrapper delivers 2–4× speed improvements over prior SWIG wrappers.
4. Learned how to seamlessly integrate classic technical indicators into a pandas DataFrame for feature engineering or backtesting.

# FinRL‑Meta Environment Layer
Explains how FinRL‑Meta builds realistic market environments for financial RL using the Gym API.

1. Layered architecture: Environment layer sits between the data layer and agent layer, enabling modular simulation of trading with transaction costs, risk controls, margin, and shorting options. Designed to plug into Gym-style RL workflows. 

2. Market realism: Incorporates market frictions like transaction costs (~0.1%), margin, and uses VIX-based risk measures for market turbulence.

3. Multiprocessing support: Leverages GPU cores (via vectorized environments like Isaac Gym) to run parallel market simulations for speed and scalability.

# OpenAI Gym & Gym Library:
A tutorial demonstrating Gym’s standard RL API, likely covering environment creation, data flow (reset, step), and common control tasks. Key points include:
1. gym.make(...) usage, and core methods: reset(), step(action), and handling observation, reward, done, and info.
2. Understanding observation_space and action_space (Discrete vs. Box) using examples like MountainCar, CartPole, LunarLander.
3. How to embed a policy in a loop:

   env = gym.make("LunarLander-v2")  
   obs, info = env.reset()  
   obs, reward, terminated, truncated, info = env.step(action)



The official Gym docs explain how Gym provides a standard RL environment API, even though Gym itself is now unmaintained; Gymnasium is its maintained successor.
* It includes instructions for creating custom environments, vectorization for parallel execution, and tutorials on Q‑learning and control tasks.
* Notes that legacy Gym doesn’t support Numpy 2.0 and recommends migrating to Gymnasium.

# WEEK - 4 : PPO/DDPG Agent Integration

# PPO

1. Explains Proximal Policy Optimization (PPO), a key actor-critic algorithm in RL.

2. Highlights how PPO uses clipped objectives to keep new policies from deviating too much from old ones—striking a balance akin to TRPO’s "trust region," but in a simpler form.

3. Mentioned that PPO is simpler to implement and tune, yet still achieves strong performance across tasks. Likely includes pseudocode or visuals showing the training loop and clipping mechanism.

4. Builds on the previous, discussing PPO in action:
* Live demo training a PPO agent on environments like CartPole or custom tasks.
* Shows reset() and step(action) loop, interaction with Gym API, logging of reward, done, etc.
* Possibly covers policy architectures (MLP, CNN) and training metrics like episode length, loss curves, rewards.






# Stable‑Baselines3 PPO Documentation
1. PPO in SB3 is based on OpenAI's PPO, borrowing core logic from Spinning Up and other implementations.
   
2. Key characteristics:
* Clipped surrogate loss for stable updates.
* Normalization of advantages, and optional clipping of value function updates. 
* Supports multi-processing via vectorized environments (e.g., make_vec_env) but no built-in recurrent nets, though recurrent versions exist in contrib.

4. Includes usage example:

   
   ``` :contentReference[oaicite:12]{index=12}
   from stable_baselines3 import PPO   
   from stable_baselines3.common.env_util import make_vec_env
   env = make_vec_env("CartPole-v1", n_envs=4)   
   model = PPO("MlpPolicy", env, verbose=1)   
   model.learn(total_timesteps=25000)   
   model.save("ppo_cartpole")
   ``` :contentReference[oaicite:12]{index=12}

# FinRL GitHub Repository

1. A full framework for financial reinforcement learning built on top of Gym and SB3.

2. Features:

* Pre-built environments simulating trading dynamics with transaction costs, slippage, portfolio constraints.
* Integrations with SB3 training pipelines—enabling usage of PPO and other algorithms on financial data.
* Tools for data ingestion, backtesting, and portfolio evaluation in a structured manner.
