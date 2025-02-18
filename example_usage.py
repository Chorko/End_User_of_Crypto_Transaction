from eth_clustering import EthereumAddressClusterer
import requests
import time
from typing import List, Dict

class AddressFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"

    def _make_api_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling
        
        Args:
            params (Dict): API request parameters
        
        Returns:
            Dict: JSON response from API or None if request fails
        """
        try:
            params['apikey'] = self.api_key
            time.sleep(0.2)  # Rate limiting
            
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == '1' or 'result' in result:
                    return result
                else:
                    print(f"API Error: {result.get('message', 'Unknown error')}")
                    return None
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

def main():
    """
    Main function to demonstrate the usage of EthereumAddressClusterer
    with dynamically fetched addresses from recent transactions
    """
    
    # Initialize with your API key
    api_key = "287QE6GERPMSSYUZ1TBMM572CJ9C9VTX4P"
    
    try:
        # Initialize address fetcher and clusterer
        fetcher = AddressFetcher(api_key)
        clusterer = EthereumAddressClusterer(api_key)
        
        # Fetch addresses from different categories
        print("Fetching addresses from recent transactions...")
        
        addresses = {
            'exchanges': fetcher.get_exchange_addresses(10),
            'defi': fetcher.get_defi_addresses(10),
            'nft': fetcher.get_nft_addresses(10),
            'individuals': fetcher.get_individual_addresses(10)
        }
        
        # Check if we got enough addresses
        total_addresses = sum(len(addrs) for addrs in addresses.values())
        if total_addresses == 0:
            raise Exception("Failed to fetch any addresses")
        
        # Flatten addresses and remove duplicates
        all_addresses = list(set([
            addr for category in addresses.values() 
            for addr in category
        ]))
        
        print(f"\nAnalyzing {len(all_addresses)} unique addresses...")
        
        # Perform clustering analysis
        print("\nStarting clustering analysis...")
        clusters = clusterer.cluster_addresses(all_addresses, eps=0.3, min_samples=3)
        
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
                    if address.lower() in [a.lower() for a in addrs]:
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
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 