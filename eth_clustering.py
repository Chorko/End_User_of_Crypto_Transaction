import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import networkx as nx
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any

class EthereumAddressClusterer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.addresses_data = {}
        self.transaction_graph = nx.Graph()
        
    def _make_api_request(self, params: Dict[str, Any]) -> Dict:
        """Make API request to Etherscan with rate limiting"""
        params['apikey'] = self.api_key
        
        # Rate limiting - Etherscan allows 5 requests per second
        time.sleep(0.2)
        
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code}")

    def fetch_transactions(self, address: str, days_back: int = 30) -> Dict:
        """Fetch both normal and internal transactions for an address"""
        end_block = 'latest'
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        # Normal transactions
        normal_tx_params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'starttime': start_timestamp,
            'endblock': end_block,
            'sort': 'asc'
        }
        
        # Internal transactions
        internal_tx_params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'starttime': start_timestamp,
            'endblock': end_block,
            'sort': 'asc'
        }
        
        normal_tx = self._make_api_request(normal_tx_params)
        internal_tx = self._make_api_request(internal_tx_params)
        
        return {
            'normal': normal_tx.get('result', []),
            'internal': internal_tx.get('result', [])
        }

    def extract_features(self, address: str, transactions: Dict) -> Dict:
        """Extract relevant features from transaction data"""
        features = {
            'total_transactions': 0,
            'total_eth_sent': 0.0,
            'total_eth_received': 0.0,
            'avg_gas_price': 0.0,
            'total_gas_spent': 0.0,
            'unique_interactions': set(),
            'transaction_frequency': 0.0
        }
        
        gas_prices = []
        timestamps = []
        
        for tx_type in ['normal', 'internal']:
            for tx in transactions[tx_type]:
                features['total_transactions'] += 1
                
                value = float(tx.get('value', 0)) / 1e18  # Convert from Wei to ETH
                
                if tx.get('from', '').lower() == address.lower():
                    features['total_eth_sent'] += value
                    features['unique_interactions'].add(tx.get('to', '').lower())
                else:
                    features['total_eth_received'] += value
                    features['unique_interactions'].add(tx.get('from', '').lower())
                
                if tx_type == 'normal':  # Gas metrics only available for normal transactions
                    gas_price = int(tx.get('gasPrice', 0))
                    gas_used = int(tx.get('gasUsed', 0))
                    gas_prices.append(gas_price)
                    features['total_gas_spent'] += (gas_price * gas_used) / 1e18
                
                timestamps.append(int(tx.get('timeStamp', 0)))
        
        if gas_prices:
            features['avg_gas_price'] = sum(gas_prices) / len(gas_prices)
        
        if timestamps:
            time_range = max(timestamps) - min(timestamps)
            features['transaction_frequency'] = features['total_transactions'] / (time_range / 86400) if time_range > 0 else 0
            
        features['unique_interactions'] = len(features['unique_interactions'])
        
        return features

    def cluster_addresses(self, addresses: List[str], eps: float = 0.5, min_samples: int = 3) -> Dict:
        """Cluster addresses based on their transaction patterns"""
        # Fetch and process data for all addresses
        for address in addresses:
            transactions = self.fetch_transactions(address)
            self.addresses_data[address] = self.extract_features(address, transactions)
        
        # Prepare feature matrix
        feature_columns = ['total_transactions', 'total_eth_sent', 'total_eth_received', 
                         'avg_gas_price', 'total_gas_spent', 'unique_interactions', 
                         'transaction_frequency']
        
        X = []
        for address in addresses:
            features = self.addresses_data[address]
            X.append([features[col] for col in feature_columns])
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X_scaled)
        
        # Organize results
        clusters = {}
        for address, label in zip(addresses, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(address)
        
        return clusters

    def build_transaction_graph(self, addresses: List[str]):
        """Build a transaction graph for visualization"""
        for address in addresses:
            transactions = self.fetch_transactions(address)
            
            for tx_type in ['normal', 'internal']:
                for tx in transactions[tx_type]:
                    from_addr = tx.get('from', '').lower()
                    to_addr = tx.get('to', '').lower()
                    value = float(tx.get('value', 0)) / 1e18
                    
                    if from_addr and to_addr:
                        if self.transaction_graph.has_edge(from_addr, to_addr):
                            self.transaction_graph[from_addr][to_addr]['weight'] += value
                        else:
                            self.transaction_graph.add_edge(from_addr, to_addr, weight=value) 