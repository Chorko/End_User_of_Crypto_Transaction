import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET() {
  try {
    // Read the latest results file
    const resultsPath = path.join(process.cwd(), "results", "latest.json")
    
    if (!fs.existsSync(resultsPath)) {
      return NextResponse.json(
        { error: "No analysis results available" },
        { status: 404 }
      )
    }

    const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"))
    return NextResponse.json(results)
  } catch (error) {
    console.error("Error reading results:", error)
    return NextResponse.json(
      { error: "Failed to read analysis results" },
      { status: 500 }
    )
  }
} 