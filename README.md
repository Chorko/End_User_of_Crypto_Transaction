# End User of Crypto Transaction

## Overview

This project provides a framework for analyzing Ethereum blockchain transactions to identify end users (individuals) and distinguish them from institutional accounts, exchanges, and smart contracts. The system uses transaction graph analysis, machine learning, and heuristic-based approaches to classify addresses and predict user behaviors.

## Features

- **Address Classification**: Categorizes Ethereum addresses into different user types (individual, institutional, exchange, DeFi user, NFT trader)
- **Transaction Graph Analysis**: Builds and analyzes networks of transactions to reveal patterns
- **Cluster Detection**: Groups similar addresses based on transaction behaviors
- **End User Identification**: Applies multiple heuristics to identify individual end users with confidence scores
- **Suspicious Activity Detection**: Flags potentially suspicious transaction patterns
- **Investment Analysis**: Identifies where users have invested their crypto assets

## End User Identification Mechanisms

The system uses several complementary approaches to identify end users:

### 1. Transaction Volume and Patterns

End users (individuals) typically exhibit distinct transaction behaviors compared to institutions or exchanges:

- **Lower Transaction Volume**: End users typically have fewer transactions (<50) compared to exchanges or institutions
- **Fewer Unique Counterparties**: Individual users interact with fewer addresses
- **Transaction Frequency**: End users often show less regular transaction patterns

### 2. Behavioral Pattern Analysis

The system detects specific behavioral patterns associated with individual users:

- **Gas Optimization**: End users tend to be more sensitive to gas prices
- **Small Token Diversity**: Individual users typically hold fewer different tokens
- **Exchange Interaction**: End users usually interact with exchanges for fiat on/off ramping
- **Time-Based Patterns**: Individual users often show distinctive transaction timing (weekly patterns, working hours activity)

### 3. Interaction Analysis

The system examines how addresses interact with known entity types:

- **Exchange Interactions**: End users typically interact with exchanges to buy/sell crypto
- **DeFi Protocol Usage**: End users engage with fewer DeFi protocols than institutional users
- **NFT Marketplaces**: Individual collectors exhibit specific patterns in NFT trading

### 4. Machine Learning Classification

Two ML models are used to improve classification accuracy:

- **Isolation Forest**: Detects anomalous addresses with unusual transaction patterns
- **Random Forest Classifier**: Predicts the most likely user category based on transaction features

### 5. End User Likelihood Score

A comprehensive scoring system calculates the probability (0-1) that an address belongs to an individual end user by combining multiple signals:

```
End User Likelihood = f(transaction_volume, network_diversity, exchange_interactions, 
                        defi_interactions, contract_status, known_patterns)
```

## Setup and Usage

### Prerequisites

- Python 3.6+
- Etherscan API key (create one at https://etherscan.io/myapikey)

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Etherscan API key:
   ```
   ETHERSCAN_API_KEY=your_api_key_here
   ```

### Running the Analysis

```python
python enduser.py
```

The script will:
1. Fetch Ethereum addresses from different categories (exchanges, DeFi, NFT, individuals)
2. Build a transaction graph connecting these addresses
3. Perform clustering analysis to group similar addresses
4. Train machine learning models on the transaction data
5. Analyze individual addresses to determine their user types
6. Generate a summary of end user analysis results

## Example Output

The system generates detailed profiles for each analyzed address, including:

- User category and confidence score
- End user likelihood score
- Investment destinations
- Suspicious activities (if any)
- Behavioral patterns
- Transaction statistics

A summary report provides aggregated statistics on identified end users.

## Future Improvements

- Integration with more blockchain data sources
- Transaction value analysis
- Gas price sensitivity analysis
- Temporal pattern analysis with transaction timestamps
- Cross-chain identity linking

