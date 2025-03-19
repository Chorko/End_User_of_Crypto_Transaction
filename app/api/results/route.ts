import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET() {
  try {
    // Read the all_results.json file instead of latest.json
    const resultsPath = path.join(process.cwd(), "results", "all_results.json")
    
    if (!fs.existsSync(resultsPath)) {
      // Fall back to latest.json if all_results.json doesn't exist yet
      const latestPath = path.join(process.cwd(), "results", "latest.json")
      
      if (!fs.existsSync(latestPath)) {
        return NextResponse.json(
          { error: "No analysis results available" },
          { status: 404 }
        )
      }
      
      const latestResults = JSON.parse(fs.readFileSync(latestPath, "utf-8"))
      return NextResponse.json([latestResults]) // Return as array for consistency
    }

    const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"))
    return NextResponse.json(results) // Already an array from our modified save_results function
  } catch (error) {
    console.error("Error reading results:", error)
    return NextResponse.json(
      { error: "Failed to read analysis results" },
      { status: 500 }
    )
  }
} 