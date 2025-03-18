import { NextResponse } from "next/server"

// Mock transactions for the API
const transactions = [
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
]

export async function GET(request: Request) {
  try {
    // In a real app, you would fetch from a database or other data source
    return NextResponse.json(transactions)
  } catch (error) {
    console.error("Error fetching transactions:", error)
    return NextResponse.json(
      { error: "Failed to fetch transactions" },
      { status: 500 }
    )
  }
} 