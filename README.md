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
- **XGBoost Model Integration**: Uses XGBoost for improved prediction accuracy
- **Comprehensive User Profiling**: Generates detailed user profiles with likelihood scores and confidence metrics
- **Detailed Event Outputs**: Provides transaction patterns and cluster analysis for each address

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

Multiple ML models are used to improve classification accuracy:

- **Isolation Forest**: Detects anomalous addresses with unusual transaction patterns
- **Random Forest Classifier**: Predicts the most likely user category based on transaction features
- **XGBoost Model**: Provides enhanced prediction capabilities with gradient boosting

### 5. End User Likelihood Score

A comprehensive scoring system calculates the probability (0-1) that an address belongs to an individual end user by combining multiple signals:

```
End User Likelihood = f(transaction_volume, network_diversity, exchange_interactions, 
                        defi_interactions, contract_status, known_patterns)
```

## Detailed Setup Instructions

### 1. Environment Setup

#### Install Python
Make sure Python 3.6 or newer is installed:
```
python --version
```

#### Create a Virtual Environment (Recommended)
```
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python -m venv venv
source venv/bin/activate
```

### 2. Project Setup

#### Clone or Download the Project
If using Git:
```
git clone https://github.com/Chorko/End_User_of_Crypto_Transaction.git
cd End_User_of_Crypto_Transaction
```

Or simply download and extract the ZIP file to a folder.

#### Install Dependencies
Install the required dependencies:
```
pip install -r requirements.txt
```

The requirements.txt file includes:
```
requests
numpy
scikit-learn
python-dotenv
networkx
matplotlib
pandas
scipy
xgboost
```

### 3. API Key Setup

#### Get an Etherscan API Key
1. Go to https://etherscan.io/
2. Register for an account if you don't have one
3. Once logged in, go to https://etherscan.io/myapikey
4. Click "Add" to create a new API key
5. Give it a name (e.g., "Transaction Analysis")
6. Copy your new API key

#### Set Up Environment Variables
Create a file named `.env` in the project root directory:
```
ETHERSCAN_API_KEY=your_api_key_here
```
Replace "your_api_key_here" with the API key from Etherscan.

### 4. Running the Project

#### Make Sure the Required Files Exist
Ensure you have the main file:
- enduser.py

#### Run the Analysis
```
python enduser.py
```

### 5. Troubleshooting Common Issues

#### API Key Issues
If you see an error about an invalid API key:
- Double-check that the `.env` file is in the correct location
- Verify the API key is correct
- Check for any spaces or typos in the `.env` file

#### Module Not Found Errors
If you see errors like "No module named 'X'":
```
pip install X
```

#### Network Connection Issues
If you see connection errors:
- Check your internet connection
- Etherscan may have rate limits - the code has rate limiting built in, but you might need to wait if you hit API limits

### 6. Verification

When the code runs successfully:
1. It will fetch Ethereum addresses from different categories
2. Build a transaction graph
3. Perform clustering analysis
4. Train machine learning models (including XGBoost)
5. Provide detailed information about analyzed addresses

You should see output showing:
- API key validation
- Address fetching
- Clustering results
- Machine learning model training
- Sample address analysis with likelihood categories (HIGH 🟢, MEDIUM 🟡, LOW 🔴)

## Example Output

The system generates detailed profiles for each analyzed address, including:

- User category and confidence score
- End user likelihood score with visual indicators (HIGH 🟢, MEDIUM 🟡, LOW 🔴)
- Profile ID and transaction patterns
- Investment destinations
- Suspicious activities (if any)
- Behavioral patterns
- Transaction statistics
- Cluster membership information

A summary report provides aggregated statistics on identified end users and saves the results to JSON files for further analysis.

## Future Improvements

- Integration with more blockchain data sources
- Transaction value analysis
- Gas price sensitivity analysis
- Temporal pattern analysis with transaction timestamps
- Cross-chain identity linking
- Web dashboard integration for interactive analysis

