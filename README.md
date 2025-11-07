AI Trading Agent using Deep Q-Learning

This project implements a reinforcement learning–based trading agent that learns to make Buy, Hold, or Sell decisions on stock market data using a **Deep Q-Network (DQN)** built with **PyTorch**.

## Overview
The agent is trained on 5 years of **AAPL** stock data (2020–2025) using technical indicators such as:
- Closing price  
- 5-day Simple Moving Average (SMA_5)  
- 20-day Simple Moving Average (SMA_20)  
- Daily returns  

After training, the agent starts with \$10,000 and achieves a **final balance of \$10,655.56** — a profit of **\$655.56**, demonstrating effective learning of trading strategies.

## Features
- Downloads stock data using **yfinance**  
- Calculates moving averages and returns for feature engineering  
- Builds a **custom trading environment** simulating portfolio management  
- Implements **Deep Q-Learning** with experience replay and epsilon-greedy exploration  
- Evaluates trading performance after training  

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/aparajita1721/AI_trading_agent.git
   cd AI_trading_agent
   ```
2. Install the dependencies:
    ```bash
   pip install -r requirements.txt
   ```
3. Run the script
   ```bash
   python trading_agent.py
   ```

   
