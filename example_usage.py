from eth_clustering import EthereumAddressClusterer

def main():
    # Initialize with your API key
    api_key = "287QE6GERPMSSYUZ1TBMM572CJ9C9VTX4P"
    clusterer = EthereumAddressClusterer(api_key)
    
    # Example addresses to analyze
    addresses = [
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Example address 1
        "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Example address 2
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Example address 3
        # Add more addresses as needed
    ]
    
    # Perform clustering
    clusters = clusterer.cluster_addresses(addresses)
    
    # Print results
    print("\nClustering Results:")
    for cluster_id, cluster_addresses in clusters.items():
        if cluster_id == -1:
            print("\nNoise/Outliers:")
        else:
            print(f"\nCluster {cluster_id}:")
        for address in cluster_addresses:
            print(f"  {address}")
            
    # Build transaction graph
    clusterer.build_transaction_graph(addresses)
    
    # You can now analyze the transaction_graph using NetworkX functions
    print("\nTransaction Graph Statistics:")
    print(f"Number of nodes: {clusterer.transaction_graph.number_of_nodes()}")
    print(f"Number of edges: {clusterer.transaction_graph.number_of_edges()}")

if __name__ == "__main__":
    main() 