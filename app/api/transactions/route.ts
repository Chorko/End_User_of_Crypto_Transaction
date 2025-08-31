import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET(request: Request) {
  try {
    // Define result paths
    const allResultsPath = path.join(process.cwd(), "results", "all_results.json")
    const latestPath = path.join(process.cwd(), "results", "latest.json")
    
    // Check if we have results data
    if (!fs.existsSync(allResultsPath) && !fs.existsSync(latestPath)) {
      return NextResponse.json(
        { error: "No analysis results available" },
        { status: 404 }
      )
    }
    
    // Read the results file
    let allResults = []
    if (fs.existsSync(allResultsPath)) {
      allResults = JSON.parse(fs.readFileSync(allResultsPath, "utf-8"))
    } else {
      // Fall back to latest.json
      const latestResults = JSON.parse(fs.readFileSync(latestPath, "utf-8"))
      allResults = [latestResults]
    }
    
    // Use the most recent results data (last item in the array)
    const latestResultData = allResults[allResults.length - 1]
    
    // Extract analyzed users and transaction data
    const analyzedUsers = latestResultData.analyzed_users || []
    
    // Generate transactions from the user data
    const transactions = []
    
    // Process each analyzed user
    for (const user of analyzedUsers) {
      const address = user.address
      
      // Skip if no address or transaction patterns
      if (!address || !user.transaction_patterns) continue
      
      // Get transaction data
      const txData = user.transaction_patterns
      
      // Add basic transaction info
      if (txData.recent_transactions && Array.isArray(txData.recent_transactions)) {
        // Use actual transaction data if available
        for (const tx of txData.recent_transactions.slice(0, 5)) { // Limit to 5 most recent
          transactions.push({
            id: tx.hash || `0x${Math.random().toString(16).substring(2, 10)}`,
            from: tx.from || address,
            to: tx.to || "0x" + Math.random().toString(16).substring(2, 42),
            amount: tx.value || ((Math.random() * 2).toFixed(4)),
            token: tx.tokenSymbol || "ETH",
            timestamp: tx.timeStamp || new Date().toISOString(),
            status: "confirmed",
            details: {
              gasUsed: tx.gasUsed || "21000",
              gasPrice: tx.gasPrice || "25 Gwei",
              blockNumber: tx.blockNumber || "12345678"
            }
          })
        }
      } else {
        // Create mock transaction data based on real user data
        // At least create one mock transaction for each analyzed address
        transactions.push({
          id: `0x${Math.random().toString(16).substring(2, 10)}`,
          from: address,
          to: Array.isArray(txData.unique_counterparties) && txData.unique_counterparties.length > 0
            ? txData.unique_counterparties[0]
            : "0x" + Math.random().toString(16).substring(2, 42),
          amount: ((Math.random() * 2).toFixed(4)),
          token: "ETH",
          timestamp: new Date().toISOString(),
          status: "confirmed",
          details: {
            gasUsed: "21000",
            gasPrice: "25 Gwei",
            blockNumber: "12345678"
          }
        })
      }
    }
    
    return NextResponse.json(transactions)
  } catch (error) {
    console.error("Error generating transactions:", error)
    return NextResponse.json(
      { error: "Failed to fetch transactions" },
      { status: 500 }
    )
  }
} 