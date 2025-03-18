export interface Transaction {
  id: string
  from: string
  to: string
  amount: string
  token: string
  timestamp: string
  status: "confirmed" | "pending" | "failed"
  details?: {
    gasUsed?: string
    gasPrice?: string
    blockNumber?: string
    nonce?: string
  }
  endUserData?: {
    address: string
    user_profile_id: number
    user_category: number
    user_category_name: string
    end_user_likelihood: number
    confidence: number
    is_anomaly: boolean
    cluster_id?: number
    behavior_patterns?: Record<string, number>
    suspicious_patterns?: string[]
  }
}

// Mock transactions data
export const mockTransactions: Transaction[] = [
  {
    id: "0x1a2b3c4d5e6f",
    from: "0x7g8h9i0j1k2l",
    to: "0x3m4n5o6p7q8r",
    amount: "1.245",
    token: "ETH",
    timestamp: "2023-03-15 14:30:45",
    status: "confirmed",
    details: {
      gasUsed: "21000",
      gasPrice: "25 Gwei",
      blockNumber: "12345678",
      nonce: "42"
    }
  },
  {
    id: "0x2b3c4d5e6f7g",
    from: "0x8h9i0j1k2l3m",
    to: "0x4n5o6p7q8r9s",
    amount: "0.75",
    token: "ETH",
    timestamp: "2023-03-15 13:25:12",
    status: "confirmed"
  },
  {
    id: "0x3c4d5e6f7g8h",
    from: "0x9i0j1k2l3m4n",
    to: "0x5o6p7q8r9s0t",
    amount: "125",
    token: "USDC",
    timestamp: "2023-03-15 12:45:30",
    status: "pending"
  },
  {
    id: "0x4d5e6f7g8h9i",
    from: "0x0j1k2l3m4n5o",
    to: "0x6p7q8r9s0t1u",
    amount: "0.15",
    token: "ETH",
    timestamp: "2023-03-15 11:30:45",
    status: "failed",
    details: {
      gasUsed: "21000",
      gasPrice: "20 Gwei",
      blockNumber: "12345677",
      nonce: "41"
    }
  },
  {
    id: "0x5e6f7g8h9i0j",
    from: "0x1k2l3m4n5o6p",
    to: "0x7q8r9s0t1u2v",
    amount: "50",
    token: "USDT",
    timestamp: "2023-03-15 10:15:00",
    status: "confirmed"
  }
];

export const mockAnalysisResults = {
  timestamp: "2023-03-15T16:30:00Z",
  total_addresses: 1245,
  total_end_users: 856,
  clusters: [
    {
      id: "0",
      size: 248,
      addresses: ["0x7g8h9i0j1k2l", "0x8h9i0j1k2l3m", "0x9i0j1k2l3m4n"]
    },
    {
      id: "1",
      size: 157,
      addresses: ["0x0j1k2l3m4n5o", "0x1k2l3m4n5o6p", "0x2l3m4n5o6p7q"]
    },
    {
      id: "2",
      size: 103,
      addresses: ["0x3m4n5o6p7q8r", "0x4n5o6p7q8r9s", "0x5o6p7q8r9s0t"]
    }
  ],
  category_distribution: {
    "Individual": 458,
    "Trader": 215,
    "Small Business": 116,
    "Developer": 67
  },
  event_outputs: [
    {
      address: "0x7g8h9i0j1k2l",
      user_profile_id: 1,
      user_category: 1,
      category: "Individual",
      end_user_likelihood: 0.92,
      confidence: 0.85,
      is_anomaly: false,
      cluster_id: 0,
      behavior_patterns: {
        "Regular Transactions": 0.78,
        "DeFi User": 0.45,
        "Low Gas Optimization": 0.32
      }
    },
    {
      address: "0x8h9i0j1k2l3m",
      user_profile_id: 2,
      user_category: 2,
      category: "Trader",
      end_user_likelihood: 0.88,
      confidence: 0.82,
      is_anomaly: false,
      cluster_id: 0,
      behavior_patterns: {
        "High Frequency": 0.91,
        "MEV Aware": 0.76,
        "Gas Optimization": 0.85
      }
    },
    {
      address: "0x9i0j1k2l3m4n",
      user_profile_id: 3,
      user_category: 3,
      category: "Small Business",
      end_user_likelihood: 0.75,
      confidence: 0.68,
      is_anomaly: false,
      cluster_id: 0,
      behavior_patterns: {
        "Regular Hours": 0.82,
        "Weekday Activity": 0.89,
        "Multiple Recipients": 0.74
      }
    },
    {
      address: "0x0j1k2l3m4n5o",
      user_profile_id: 4,
      user_category: 1,
      category: "Individual",
      end_user_likelihood: 0.65,
      confidence: 0.58,
      is_anomaly: true,
      cluster_id: 1,
      behavior_patterns: {
        "Irregular Timing": 0.78,
        "High Value Transfers": 0.65
      },
      suspicious_patterns: ["Unusual Transfer Pattern", "Rapid Exchanges"]
    },
    {
      address: "0x1k2l3m4n5o6p",
      user_profile_id: 5,
      user_category: 4,
      category: "Developer",
      end_user_likelihood: 0.95,
      confidence: 0.91,
      is_anomaly: false,
      cluster_id: 1,
      behavior_patterns: {
        "Contract Interaction": 0.95,
        "Testing Pattern": 0.88,
        "Gas Optimization": 0.92
      }
    }
  ]
}; 