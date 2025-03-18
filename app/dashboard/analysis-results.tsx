"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"

interface AnalysisResults {
  timestamp: string
  total_addresses: number
  clusters: Array<{
    id: string
    size: number
    addresses: string[]
  }>
  category_distribution: Record<string, number>
  visualization_data: {
    nodes: Array<{
      id: string
      group: number
      value: number
      label: string
    }>
    links: Array<{
      source: string
      target: string
      value: number
    }>
  }
}

export function AnalysisResults() {
  const [results, setResults] = useState<AnalysisResults | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await fetch('/api/results')
        if (!response.ok) {
          throw new Error('Failed to fetch results')
        }
        const data = await response.json()
        setResults(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load results')
      } finally {
        setIsLoading(false)
      }
    }

    fetchResults()
    // Refresh every 30 seconds
    const interval = setInterval(fetchResults, 30000)
    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Analysis Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-4 w-[250px]" />
            <Skeleton className="h-4 w-[200px]" />
            <Skeleton className="h-4 w-[300px]" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Analysis Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-red-500">Error: {error}</div>
        </CardContent>
      </Card>
    )
  }

  if (!results) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Analysis Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div>No analysis results available</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card>
        <CardHeader>
          <CardTitle>Analysis Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Summary Section */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Summary</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Total Addresses</p>
                  <p className="text-2xl font-bold">{results.total_addresses}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Clusters</p>
                  <p className="text-2xl font-bold">{results.clusters.length}</p>
                </div>
              </div>
            </div>

            {/* Category Distribution */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Category Distribution</h3>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(results.category_distribution).map(([category, count]) => (
                  <div key={category} className="flex items-center justify-between">
                    <span className="text-sm">{category}</span>
                    <Badge variant="secondary">{count}</Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Clusters */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Clusters</h3>
              <div className="space-y-4">
                {results.clusters.map((cluster) => (
                  <div key={cluster.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">Cluster {cluster.id}</span>
                      <Badge>{cluster.size} addresses</Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Sample addresses: {cluster.addresses.slice(0, 3).join(", ")}
                      {cluster.addresses.length > 3 && "..."}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Last Updated */}
            <div className="text-sm text-muted-foreground">
              Last updated: {new Date(results.timestamp).toLocaleString()}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
} 