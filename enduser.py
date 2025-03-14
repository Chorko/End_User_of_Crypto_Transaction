from oldfunc import EthereumAddressClusterer
import requests
import time
import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

class AddressFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests

    def _make_api_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling
        
        Args:
            params (Dict): API request parameters
        
        Returns:
            Dict: JSON response from API or None if request fails
        """
        try:
            # Ensure rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            params['apikey'] = self.api_key
            
            response = requests.get(self.base_url, params=params)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == '1' or 'result' in result:
                    return result
                elif result.get('message') == 'NOTOK' and 'Invalid API Key' in result.get('result', ''):
                    raise Exception("Invalid API Key. Please check your Etherscan API key.")
                else:
                    print(f"API Error: {result.get('message', 'Unknown error')}")
                    return None
            elif response.status_code == 429:
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(1)  # Wait longer on rate limit
                return self._make_api_request(params)  # Retry
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
            
        except Exception as e:
            print(f"Request Error: {str(e)}")
            return None

    def get_exchange_addresses(self, count: int = 10) -> List[str]:
        """Get addresses from major exchange transactions"""
        # Known exchange addresses to start with
        exchanges = [
            "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance
            "0x2910543af39aba0cd09dbb2d50200b3e800a63d2"   # Kraken
        ]
        
        addresses = set()
        for exchange in exchanges:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': exchange,
                'startblock': 'latest',
                'page': 1,
                'offset': 20,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    addresses.add(tx['from'])
                    addresses.add(tx['to'])
                    if len(addresses) >= count:
                        break
        
        return list(addresses)[:count]

    def get_defi_addresses(self, count: int = 10) -> List[str]:
        """Get addresses interacting with DeFi protocols"""
        # Popular DeFi contract addresses
        defi_contracts = [
            "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",  # Aave
            "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45"   # Uniswap
        ]
        
        addresses = set()
        for contract in defi_contracts:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': contract,
                'startblock': 'latest',
                'page': 1,
                'offset': 20,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    addresses.add(tx['from'])
                    if len(addresses) >= count:
                        break
        
        return list(addresses)[:count]

    def get_nft_addresses(self, count: int = 10) -> List[str]:
        """Get addresses from NFT marketplace transactions"""
        # Popular NFT marketplace contracts
        nft_contracts = [
            "0x7Be8076f4EA4A4AD08075C2508e481d6C946D12b",  # OpenSea
            "0x74312363e45dcaba76c59ec49a7aa8a65a67eed3"   # X2Y2
        ]
        
        addresses = set()
        for contract in nft_contracts:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': contract,
                'startblock': 'latest',
                'page': 1,
                'offset': 20,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    addresses.add(tx['from'])
                    if len(addresses) >= count:
                        break
        
        return list(addresses)[:count]

    def get_individual_addresses(self, count: int = 10) -> List[str]:
        """Get addresses from recent normal transactions"""
        # Get transactions from a recent block
        params = {
            'module': 'proxy',  # Using proxy API for latest blocks
            'action': 'eth_getBlockByNumber',
            'tag': 'latest',
            'boolean': 'true'
        }
        
        addresses = set()
        response = self._make_api_request(params)
        
        if response and 'result' in response and 'transactions' in response['result']:
            for tx in response['result']['transactions']:
                # Filter out contract creations and known patterns
                if 'to' in tx and tx['to'] and not tx['to'].startswith('0x000000'):
                    addresses.add(tx['from'].lower())
                    if len(addresses) >= count:
                        break
        
        # If we need more addresses, fetch from normal transactions
        if len(addresses) < count:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': '0x0000000000000000000000000000000000000000',  # Use null address as reference
                'startblock': 'latest',
                'page': 1,
                'offset': 50,
                'sort': 'desc'
            }
            
            response = self._make_api_request(params)
            if response and 'result' in response:
                for tx in response['result']:
                    if 'to' in tx and tx['to'] and not tx['to'].startswith('0x000000'):
                        addresses.add(tx['from'].lower())
                        if len(addresses) >= count:
                            break
        
        return list(addresses)[:count]

class EndUserPredictor:
    def __init__(self, transaction_graph, clusters):
        self.transaction_graph = transaction_graph
        self.clusters = clusters
        self.isolation_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
        # Define user categories
        self.user_categories = {
            0: "Individual/Retail User",
            1: "Institutional/Large Investor",
            2: "Exchange/Protocol Account",
            3: "DeFi User",
            4: "NFT Trader/Collector"
        }
        
        # Known DeFi protocol contract addresses
        self.defi_contracts = {
            "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave Lending Pool",
            "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": "Aave Token",
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "Uniswap Token",
            "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "Uniswap Router",
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
            "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
            "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
            "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
            "0x1985365e9f78359a9b6ad760e32412f4a445e862": "REP",
            "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": "MKR",
            "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e": "YFI",
            "0xba100000625a3754423978a60c9317c58a424e3d": "BAL",
            "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f": "SNX"
        }
        
        # Known NFT marketplace contract addresses
        self.nft_contracts = {
            "0x7be8076f4ea4a4ad08075c2508e481d6c946d12b": "OpenSea",
            "0x60e4d786628fea6478f785a6d7e704777c86a7c6": "MAYC",
            "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d": "BAYC",
            "0x34d85c9cdeb23fa97cb08333b511ac86e1c4e258": "Otherdeed",
            "0x74312363e45dcaba76c59ec49a7aa8a65a67eed3": "X2Y2",
            "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb": "CryptoPunks",
            "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85": "ENS"
        }
        
        # Known exchange addresses
        self.exchange_addresses = {
            "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
            "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
            "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "Gate.io",
            "0x11111112542d85b3ef69ae05771c2dccff4faa26": "1inch",
            "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
            "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "KuCoin",
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance Hot Wallet"
        }
        
        # Suspicious patterns
        self.suspicious_patterns = [
            "high_velocity_small_amounts",  # Many small transactions in short time
            "tornado_cash_interaction",     # Interaction with mixers
            "wash_trading",                 # Trading same assets back and forth
            "chain_hopping",                # Moving across multiple chains
            "dormant_reactivation"          # Long dormant account suddenly active
        ]
        
        # Cluster explanation map
        self.cluster_explanations = {
            -1: "Outlier/Unclustered (This address has unique transaction patterns that don't match any cluster)",
            0: "Primary Activity Cluster (High transaction volume with diverse counterparties)",
            1: "Exchange-Related Cluster (Addresses likely related to exchange activity)",
            2: "DeFi Activity Cluster (Addresses engaging with DeFi protocols)",
            3: "NFT Trading Cluster (Addresses involved in NFT marketplace activity)",
            4: "Mixed Usage Cluster (Addresses with multiple types of activity)"
        }
        
    def extract_features(self, addresses: List[str]) -> np.ndarray:
        """Extract numerical features for each address"""
        features = []
        for address in addresses:
            if address in self.transaction_graph:
                # Node-level features
                degree = self.transaction_graph.degree(address)
                
                # Check if graph is directed before using in_degree and out_degree
                if hasattr(self.transaction_graph, 'is_directed') and self.transaction_graph.is_directed():
                    in_degree = self.transaction_graph.in_degree(address)
                    out_degree = self.transaction_graph.out_degree(address)
                else:
                    # For undirected graphs, in_degree = out_degree = degree
                    in_degree = degree
                    out_degree = degree
                
                # Get cluster information
                cluster_id = -1
                for cid, cluster_addrs in self.clusters.items():
                    if address in cluster_addrs:
                        cluster_id = cid
                        break
                
                # Create feature vector
                feature_vector = [
                    degree,
                    in_degree,
                    out_degree,
                    cluster_id,
                    len(list(self.transaction_graph.neighbors(address)))
                ]
                features.append(feature_vector)
            else:
                features.append([0, 0, 0, -1, 0])  # Default features for unknown addresses
                
        return np.array(features)
    
    def categorize_addresses(self, addresses: List[str]) -> List[int]:
        """
        Categorize addresses based on transaction patterns and characteristics
        Returns list of category IDs corresponding to self.user_categories
        """
        categories = []
        features = self.extract_features(addresses)
        
        for i, address in enumerate(addresses):
            if address not in self.transaction_graph:
                categories.append(0)  # Default to Individual/Retail User
                continue
                
            # Extract basic metrics
            degree = self.transaction_graph.degree(address)
            neighbors = list(self.transaction_graph.neighbors(address))
            
            # Find address cluster
            cluster_id = -1
            for cid, cluster_addrs in self.clusters.items():
                if address in cluster_addrs:
                    cluster_id = cid
                    break
            
            # Check if address is lower or uppercase (exchanges often use checksum addresses)
            is_checksum = any(c.isupper() for c in address[2:])
            
            # Categorization logic with more refined thresholds
            if self._is_exchange_account(address, neighbors) or degree > 500:
                # High transaction volume indicates exchange or protocol
                categories.append(2)  # Exchange/Protocol Account
            elif degree > 100 and cluster_id >= 0 and len(self.clusters.get(cluster_id, [])) > 5:
                # Significant transaction volume in a large cluster
                categories.append(1)  # Institutional/Large Investor
            elif self._is_likely_defi_user(address, neighbors):
                # Interacts with known DeFi contracts
                categories.append(3)  # DeFi User
            elif self._is_likely_nft_trader(address, neighbors):
                # Interacts with known NFT marketplaces
                categories.append(4)  # NFT Trader/Collector
            else:
                # Default category
                categories.append(0)  # Individual/Retail User
                
        return categories
    
    def _is_likely_defi_user(self, address: str, neighbors: List[str]) -> bool:
        """Determine if address is likely a DeFi user based on its interactions"""
        # Check if address interacts with DeFi protocols
        for neighbor in neighbors:
            if neighbor.lower() in self.defi_contracts:
                return True
                
        # Check for partial matches (smart contract might have different implementations)
        for neighbor in neighbors:
            for defi_addr in self.defi_contracts:
                if neighbor.lower().startswith(defi_addr[:10].lower()):
                    return True
        
        return False
    
    def _is_likely_nft_trader(self, address: str, neighbors: List[str]) -> bool:
        """Determine if address is likely an NFT trader based on its interactions"""
        # Check if address interacts with NFT marketplaces
        for neighbor in neighbors:
            if neighbor.lower() in self.nft_contracts:
                return True
                
        # Check for partial matches
        for neighbor in neighbors:
            for nft_addr in self.nft_contracts:
                if neighbor.lower().startswith(nft_addr[:10].lower()):
                    return True
        
        return False
    
    def _is_exchange_account(self, address: str, neighbors: List[str]) -> bool:
        """Determine if address is likely an exchange account based on interactions"""
        # Check if address interacts with known exchanges
        for neighbor in neighbors:
            if neighbor.lower() in self.exchange_addresses:
                return True
                
        # Check for partial matches
        for neighbor in neighbors:
            for exchange_addr in self.exchange_addresses:
                if neighbor.lower().startswith(exchange_addr[:10].lower()):
                    return True
        
        return False
    
    def detect_investment_destinations(self, address: str) -> Dict[str, float]:
        """
        Identify where the user has invested their crypto
        
        Returns:
            Dict[str, float]: Dictionary mapping investment destinations to 
                              estimated portion of portfolio (0-1 scale)
        """
        if address not in self.transaction_graph:
            return {}
            
        neighbors = list(self.transaction_graph.neighbors(address))
        investment_map = {}
        total_weight = 0
        
        # Check DeFi investments
        for neighbor in neighbors:
            neighbor_lower = neighbor.lower()
            
            # Check DeFi protocols
            for contract, protocol_name in self.defi_contracts.items():
                if neighbor_lower.startswith(contract[:10].lower()):
                    weight = 1.0 / len(neighbors)  # Simplified weighting
                    if protocol_name in investment_map:
                        investment_map[protocol_name] += weight
                    else:
                        investment_map[protocol_name] = weight
                    total_weight += weight
                    break
                    
            # Check NFT investments
            for contract, marketplace_name in self.nft_contracts.items():
                if neighbor_lower.startswith(contract[:10].lower()):
                    weight = 1.0 / len(neighbors)
                    if marketplace_name in investment_map:
                        investment_map[marketplace_name] += weight
                    else:
                        investment_map[marketplace_name] = weight
                    total_weight += weight
                    break
                    
            # Check exchange deposits
            for exchange_addr, exchange_name in self.exchange_addresses.items():
                if neighbor_lower.startswith(exchange_addr[:10].lower()):
                    weight = 1.0 / len(neighbors)
                    if exchange_name in investment_map:
                        investment_map[exchange_name] += weight
                    else:
                        investment_map[exchange_name] = weight
                    total_weight += weight
                    break
        
        # Normalize weights
        if total_weight > 0:
            for dest in investment_map:
                investment_map[dest] /= total_weight
                
        # Sort by weight descending
        return dict(sorted(investment_map.items(), key=lambda x: x[1], reverse=True))
    
    def detect_suspicious_activities(self, address: str) -> List[Dict[str, any]]:
        """
        Detect potential suspicious activities for the given address
        
        Returns:
            List[Dict[str, any]]: List of suspicious patterns detected with confidence scores
        """
        if address not in self.transaction_graph:
            return []
            
        suspicious_activities = []
        neighbors = list(self.transaction_graph.neighbors(address))
        
        # Check for high velocity of transactions - only flag for relatively high volumes
        degree = self.transaction_graph.degree(address)
        if degree > 100:  # Increased threshold from 50 to 100
            suspicious_activities.append({
                "pattern": "high_velocity_small_amounts",
                "confidence": min(0.8, degree / 1000),  # Capped at 0.8, more gradual increase
                "description": "High volume of transactions detected, possibly automated trading or suspicious activity"
            })
            
        # Check for interaction with known mixing services
        tornado_cash_addresses = [
            "0x722122df12d4e14e13ac3b6895a86e84145b6967",
            "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
            "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
            "0xa160cdab225685da1d56aa342ad8841c3b53f291"
        ]
        
        for neighbor in neighbors:
            if any(neighbor.lower().startswith(mixer[:10].lower()) for mixer in tornado_cash_addresses):
                suspicious_activities.append({
                    "pattern": "tornado_cash_interaction",
                    "confidence": 0.9,
                    "description": "Interaction with privacy mixer detected, possibly attempting to obscure transaction history"
                })
                break
                
        # Check for potential wash trading - now requires bidirectional transactions and at least 3 transactions
        # This reduces false positives for addresses with just 1-2 transactions
        bidirectional_neighbors = [n for n in neighbors if address in self.transaction_graph.neighbors(n)]
        if len(bidirectional_neighbors) > 0 and degree >= 3:
            confidence = min(0.7, len(bidirectional_neighbors) / len(neighbors) * 0.7)
            suspicious_activities.append({
                "pattern": "wash_trading",
                "confidence": confidence,
                "description": "Potential wash trading detected, trading with the same addresses repeatedly"
            })
            
        # Dormant reactivation would require historical data, this is a placeholder
        # In a real implementation, you would check the timestamp of transactions
        
        return suspicious_activities
    
    def get_cluster_explanation(self, cluster_id: int) -> str:
        """Get human-readable explanation for a cluster ID"""
        return self.cluster_explanations.get(cluster_id, f"Cluster {cluster_id} (No specific information available)")
    
    def calculate_confidence_adjustment(self, prediction: int, address: str) -> float:
        """
        Calculate a confidence adjustment based on the quality of evidence
        
        Args:
            prediction: The predicted category ID
            address: The Ethereum address
            
        Returns:
            float: A confidence adjustment factor (0.0 to 1.0)
        """
        if address not in self.transaction_graph:
            return 0.3  # Low confidence for addresses not in the graph
        
        neighbors = list(self.transaction_graph.neighbors(address))
        degree = self.transaction_graph.degree(address)
        
        # Start with base confidence
        adjustment = 0.5
        
        # Find cluster information
        cluster_id = -1
        for cid, cluster_addrs in self.clusters.items():
            if address in cluster_addrs:
                cluster_id = cid
                break
        
        # More transactions = stronger evidence
        if degree > 100:
            adjustment += 0.2
        elif degree > 10:
            adjustment += 0.1
        elif degree < 3:
            adjustment -= 0.2  # Very few transactions = low confidence
            
        # Being in a cluster increases confidence (except for outliers)
        if cluster_id >= 0:
            adjustment += 0.1
        else:
            adjustment -= 0.1  # Outliers get reduced confidence
            
        # Strong evidence for specific categories
        if prediction == 2 and self._is_exchange_account(address, neighbors):  # Exchange
            adjustment += 0.2
        elif prediction == 3 and self._is_likely_defi_user(address, neighbors):  # DeFi
            adjustment += 0.2
        elif prediction == 4 and self._is_likely_nft_trader(address, neighbors):  # NFT
            adjustment += 0.2
            
        # Cap adjustment between 0.3 and 1.0
        return max(0.3, min(1.0, adjustment))
    
    def identify_end_user(self, address: str) -> Dict[str, any]:
        """
        Identify detailed information about the end user of this address
        
        Returns:
            Dict[str, any]: Detailed user profile including category, investments,
                           suspicious activities, and transaction patterns
        """
        if address not in self.transaction_graph:
            return {"address": address, "error": "Address not found in transaction graph"}
            
        # Get basic user classification
        user_profile = self.predict_user_type(address)
        
        # Add investment destinations
        user_profile["investments"] = self.detect_investment_destinations(address)
        
        # Add suspicious activities
        user_profile["suspicious_activities"] = self.detect_suspicious_activities(address)
        
        # Add transaction patterns
        neighbors = list(self.transaction_graph.neighbors(address))
        
        # Find cluster ID
        cluster_id = -1
        for cid, cluster_addrs in self.clusters.items():
            if address in cluster_addrs:
                cluster_id = cid
                break
                
        user_profile["transaction_patterns"] = {
            "total_transactions": self.transaction_graph.degree(address),
            "unique_counterparties": len(neighbors),
            "cluster_id": cluster_id,
            "cluster_explanation": self.get_cluster_explanation(cluster_id)
        }
        
        # Generate a unique identifier based on behavioral patterns
        profile_features = [
            user_profile["user_category"],
            len(user_profile["investments"]),
            len(user_profile["suspicious_activities"]),
            user_profile["transaction_patterns"]["total_transactions"]
        ]
        user_profile["user_profile_id"] = hash(tuple(profile_features)) % 10000000
        
        return user_profile
    
    def train_isolation_forest(self, addresses: List[str]):
        """Train Isolation Forest for anomaly detection"""
        features = self.extract_features(addresses)
        features_scaled = self.scaler.fit_transform(features)
        
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(features_scaled)
    
    def train_random_forest(self, addresses: List[str]):
        """Train Random Forest for user classification"""
        # Generate meaningful labels instead of random ones
        labels = self.categorize_addresses(addresses)
        
        features = self.extract_features(addresses)
        features_scaled = self.scaler.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42
        )
        
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.random_forest.fit(X_train, y_train)
        
    def predict_user_type(self, address: str) -> Dict[str, any]:
        """Combine all models to predict user type and characteristics"""
        features = self.extract_features([address])
        features_scaled = self.scaler.transform(features)
        
        results = {
            "address": address,
            "is_anomaly": False,
            "user_category": None,
            "user_category_name": None,
            "confidence": 0.0
        }
        
        # Anomaly detection
        if self.isolation_forest is not None:
            anomaly_score = self.isolation_forest.score_samples(features_scaled)[0]
            results["is_anomaly"] = anomaly_score < -0.5  # Threshold can be adjusted
            results["anomaly_score"] = anomaly_score
            
        # User category prediction
        if self.random_forest is not None:
            prediction = self.random_forest.predict(features_scaled)[0]
            probabilities = self.random_forest.predict_proba(features_scaled)[0]
            
            # Get base confidence from model
            base_confidence = max(probabilities)
            
            # Apply evidence-based adjustment
            confidence_adjustment = self.calculate_confidence_adjustment(prediction, address)
            
            # Final confidence is base * adjustment, capped at 0.95
            final_confidence = min(0.95, base_confidence * confidence_adjustment)
            
            results["user_category"] = prediction
            results["user_category_name"] = self.user_categories.get(prediction, "Unknown")
            results["confidence"] = final_confidence
            results["confidence_factors"] = {
                "model_confidence": base_confidence,
                "evidence_adjustment": confidence_adjustment
            }
            
        return results

def main():
    """
    Main function to demonstrate the usage of EthereumAddressClusterer
    with dynamically fetched addresses and end user prediction
    """
    
    # Get API key from environment variable
    api_key = os.getenv('ETHERSCAN_API_KEY')
    if not api_key:
        print("Error: ETHERSCAN_API_KEY environment variable not set")
        print("Please create a .env file with your Etherscan API key:")
        print("ETHERSCAN_API_KEY=your_api_key_here")
        return
    
    try:
        # Test API key validity first
        fetcher = AddressFetcher(api_key)
        test_params = {
            'module': 'account',
            'action': 'balance',
            'address': '0x0000000000000000000000000000000000000000'
        }
        if not fetcher._make_api_request(test_params):
            print("Failed to validate API key. Please check your Etherscan API key.")
            return
            
        print("API key validated successfully!")
        
        # Initialize clusterer
        clusterer = EthereumAddressClusterer(api_key)
        
        # Fetch addresses from different categories
        print("\nFetching addresses from recent transactions...")
        
        addresses = {
            'exchanges': fetcher.get_exchange_addresses(5),  # Reduced count for testing
            'defi': fetcher.get_defi_addresses(5),
            'nft': fetcher.get_nft_addresses(5),
            'individuals': fetcher.get_individual_addresses(5)
        }
        
        # Check if we got enough addresses
        total_addresses = sum(len(addrs) for addrs in addresses.values())
        if total_addresses == 0:
            raise Exception("Failed to fetch any addresses")
        
        # Flatten addresses and remove duplicates
        all_addresses = list(set([
            addr for category in addresses.values() 
            for addr in category if addr  # Filter out None values
        ]))
        
        print(f"\nAnalyzing {len(all_addresses)} unique addresses...")
        
        # Perform clustering analysis with more conservative parameters
        print("\nStarting clustering analysis...")
        clusters = clusterer.cluster_addresses(all_addresses, eps=0.5, min_samples=2)
        
        # Print clustering results with category labels
        print("\nClustering Results:")
        for cluster_id, cluster_addresses in clusters.items():
            if cluster_id == -1:
                print("\nNoise/Outliers (addresses that don't fit in any cluster):")
            else:
                print(f"\nCluster {cluster_id}:")
            
            for address in cluster_addresses:
                category = "Unknown"
                for cat, addrs in addresses.items():
                    if address and addrs and address.lower() in [a.lower() for a in addrs if a]:
                        category = cat.capitalize()
                        break
                print(f"  {address} ({category})")
        
        # Build and analyze transaction graph
        print("\nBuilding transaction graph...")
        clusterer.build_transaction_graph(all_addresses)
        
        # Print graph statistics
        print("\nTransaction Graph Statistics:")
        print(f"Number of unique addresses (nodes): {clusterer.transaction_graph.number_of_nodes()}")
        print(f"Number of transactions (edges): {clusterer.transaction_graph.number_of_edges()}")
        
        # After clustering analysis, initialize and train the predictor
        print("\nInitializing End User Predictor...")
        predictor = EndUserPredictor(clusterer.transaction_graph, clusters)
        
        # Use address list from graph
        address_list = list(clusterer.transaction_graph.nodes())
        
        print("\nTraining machine learning models...")
        predictor.train_isolation_forest(address_list)
        predictor.train_random_forest(address_list)  # No need to pass labels, method generates them
        
        print("\nAnalyzing sample addresses...")
        for address in address_list[:5]:  # Analyze first 5 addresses as example
            # Get detailed user profile
            user_profile = predictor.identify_end_user(address)
            
            # Print detailed information
            print(f"\n{'='*60}")
            print(f"ADDRESS: {address}")
            print(f"{'='*60}")
            print(f"USER CATEGORY: {user_profile['user_category_name']} (Category {user_profile['user_category']})")
            print(f"CONFIDENCE: {user_profile['confidence']:.2f}")
            print(f"PROFILE ID: {user_profile.get('user_profile_id', 'N/A')}")
            print(f"ANOMALY: {'Yes' if user_profile['is_anomaly'] else 'No'}")
            
            # Print investment destinations
            if user_profile.get('investments'):
                print("\nINVESTMENT DESTINATIONS:")
                for destination, weight in user_profile['investments'].items():
                    print(f"  - {destination}: {weight:.2%}")
            else:
                print("\nINVESTMENT DESTINATIONS: None detected")
                
            # Print suspicious activities
            if user_profile.get('suspicious_activities'):
                print("\nSUSPICIOUS ACTIVITIES:")
                for activity in user_profile['suspicious_activities']:
                    print(f"  - {activity['description']} (Confidence: {activity['confidence']:.2f})")
            else:
                print("\nSUSPICIOUS ACTIVITIES: None detected")
                
            # Print transaction patterns
            if user_profile.get('transaction_patterns'):
                print("\nTRANSACTION PATTERNS:")
                patterns = user_profile['transaction_patterns']
                print(f"  - Total Transactions: {patterns.get('total_transactions', 'N/A')}")
                print(f"  - Unique Counterparties: {patterns.get('unique_counterparties', 'N/A')}")
                cluster_id = patterns.get('cluster_id', 'N/A')
                print(f"  - Cluster ID: {cluster_id}")
                print(f"  - Cluster Info: {patterns.get('cluster_explanation', 'N/A')}")
            
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 