"use client"

import { useState, useEffect } from "react"
import { ArrowUpDown, ChevronLeft, ChevronRight, ExternalLink, ChevronDown, ChevronUp } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// Helper function to get colors based on category
function getCategoryColor(category: number): string {
  switch (category) {
    case 0: return "border-indigo-500 text-indigo-500 bg-indigo-500/10";
    case 1: return "border-pink-500 text-pink-500 bg-pink-500/10";
    case 2: return "border-purple-500 text-purple-500 bg-purple-500/10";
    case 3: return "border-teal-500 text-teal-500 bg-teal-500/10";
    case 4: return "border-cyan-500 text-cyan-500 bg-cyan-500/10";
    default: return "border-gray-500 text-gray-500 bg-gray-500/10";
  }
}

// Define the Transaction interface to replace the import
interface Transaction {
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

interface TransactionRowProps {
  transaction: Transaction
  isLoading?: boolean
  expanded?: boolean
  onExpandToggle?: () => void
}

function TransactionRow({ transaction, isLoading = false, expanded = false, onExpandToggle = () => {} }: TransactionRowProps) {
  if (isLoading) {
    return (
      <TableRow>
        <TableCell><Skeleton className="h-4 w-20" /></TableCell>
        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
        <TableCell><Skeleton className="h-4 w-16" /></TableCell>
        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
        <TableCell className="text-center"><Skeleton className="h-4 w-16 mx-auto" /></TableCell>
        <TableCell className="text-right"></TableCell>
      </TableRow>
    )
  }

  return (
    <>
      <TableRow className={expanded ? "bg-secondary/30" : undefined}>
        <TableCell className="font-mono text-xs">
          {transaction.id.substring(0, 10)}...
        </TableCell>
        <TableCell className="font-mono text-xs">
          <div className="flex flex-col">
            <span>{transaction.from.substring(0, 8)}...</span>
            {transaction.endUserData && (
              <Badge 
                variant={transaction.endUserData.end_user_likelihood > 0.7 ? "default" : "outline"} 
                className={`mt-1 w-fit ${getCategoryColor(transaction.endUserData.user_category)}`}
              >
                {transaction.endUserData.user_category_name || 
                 (transaction.endUserData.user_category === 0 ? "Individual" : 
                  transaction.endUserData.user_category === 1 ? "Institutional" :
                  transaction.endUserData.user_category === 2 ? "Exchange" :
                  transaction.endUserData.user_category === 3 ? "DeFi User" :
                  transaction.endUserData.user_category === 4 ? "NFT Trader" : "Unknown")}
              </Badge>
            )}
          </div>
        </TableCell>
        <TableCell className="font-mono text-xs">
          {transaction.to.substring(0, 8)}...
        </TableCell>
        <TableCell>
          <span className="font-medium">{transaction.amount}</span>
          <span className="ml-1 text-xs text-muted-foreground">{transaction.token}</span>
        </TableCell>
        <TableCell className="text-sm">{transaction.timestamp}</TableCell>
        <TableCell className="text-center">
          <Badge
            variant="outline"  // Use consistent outline variant for all statuses
            className={`capitalize ${
              transaction.status === "confirmed" ? "bg-green-500/20 text-green-500 border-green-500/50 hover:bg-green-500/20" : 
              transaction.status === "pending" ? "bg-yellow-500/20 text-yellow-500 border-yellow-500/50 hover:bg-yellow-500/20" :
              "bg-red-500/20 text-red-500 border-red-500/50 hover:bg-red-500/20"
            }`}
          >
            {transaction.status}
          </Badge>
        </TableCell>
        <TableCell className="text-right" onClick={(e) => {
          e.stopPropagation(); // Prevent row click from triggering when clicking the button
          onExpandToggle();
        }}>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>
        </TableCell>
      </TableRow>
      
      {expanded && (
        <TableRow className="bg-secondary/20">
          <TableCell colSpan={7} className="py-4">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <div>
                  <div className="text-xs font-medium text-muted-foreground">Gas Used</div>
                  <div>{transaction.details?.gasUsed || "N/A"}</div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground">Gas Price</div>
                  <div>{transaction.details?.gasPrice || "N/A"}</div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground">Block</div>
                  <div>{transaction.details?.blockNumber || "N/A"}</div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground">Nonce</div>
                  <div>{transaction.details?.nonce || "N/A"}</div>
                </div>
              </div>
              
              {transaction.endUserData && (
                <div className="mt-4 p-3 border border-green-900/30 bg-green-950/20 rounded-md">
                  <h4 className="text-sm font-medium mb-2">End User Analysis</h4>
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                    <div>
                      <div className="text-xs font-medium text-muted-foreground">User Category</div>
                      <div className="font-medium">{transaction.endUserData.user_category_name}</div>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-muted-foreground">End User Likelihood</div>
                      <div className="font-medium">
                        {(transaction.endUserData.end_user_likelihood * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-muted-foreground">Confidence</div>
                      <div className="font-medium">
                        {(transaction.endUserData.confidence * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-muted-foreground">Cluster ID</div>
                      <div className="font-medium">
                        {transaction.endUserData.cluster_id !== undefined 
                          ? transaction.endUserData.cluster_id 
                          : "N/A"}
                      </div>
                    </div>
                  </div>
                  
                  {transaction.endUserData.is_anomaly && (
                    <div className="mt-2 p-2 bg-red-950/30 border border-red-900/30 rounded-md text-xs">
                      <span className="font-semibold text-red-400">Warning:</span> Anomalous behavior detected
                    </div>
                  )}
                  
                  {transaction.endUserData.behavior_patterns && Object.keys(transaction.endUserData.behavior_patterns).length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs font-medium text-muted-foreground mb-1">Behavior Patterns</div>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(transaction.endUserData.behavior_patterns).map(([pattern, score]) => (
                          <Badge key={pattern} variant="secondary" className="text-xs">
                            {pattern}: {(score * 100).toFixed(0)}%
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              <div className="flex justify-end">
                <Button variant="outline" size="sm" className="gap-1">
                  <ExternalLink className="h-3.5 w-3.5" />
                  <span>View on Explorer</span>
                </Button>
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

const TableSkeleton = () => (
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <Skeleton className="h-8 w-[250px]" />
      <Skeleton className="h-8 w-[120px]" />
    </div>
    <div className="border rounded-md">
      <div className="h-12 px-4 border-b flex items-center">
        <Skeleton className="h-4 w-full" />
      </div>
      {Array(5).fill(null).map((_, i) => (
        <div key={i} className="h-16 px-4 border-b flex items-center">
          <Skeleton className="h-8 w-full" />
        </div>
      ))}
    </div>
    <div className="flex items-center justify-between">
      <Skeleton className="h-8 w-[100px]" />
      <Skeleton className="h-8 w-[200px]" />
    </div>
  </div>
)

export function TransactionTable({ 
  showEndUserInfo = false, 
  filterByEndUsers = false, 
  showPagination = true, 
  isLoading = false 
}: {
  showEndUserInfo?: boolean
  filterByEndUsers?: boolean
  showPagination?: boolean
  isLoading?: boolean
}) {
  const [page, setPage] = useState(1)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [sort, setSort] = useState<{ column: keyof Transaction | "endUserLikelihood" | null; direction: "asc" | "desc" }>({
    column: null,
    direction: "desc",
  })
  
  // Add a loading state
  const [loadingData, setLoadingData] = useState(true)
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([])
  
  // Add itemsPerPage constant
  const itemsPerPage = 10
  
  // Update useEffect to fetch transactions and end user data
  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        const response = await fetch('/api/results')
        if (!response.ok) {
          throw new Error('Failed to fetch transactions')
        }
        const allResults = await response.json()
        
        // Use the most recent result
        const latestResult = allResults[allResults.length - 1]
        
        // Convert event outputs to transactions
        const transactions = latestResult.event_outputs?.map((event: any) => ({
          id: event.address,
          from: event.address,
          to: event.address, // Since we don't have actual transaction data, use same address
          amount: event.total_transactions?.toString() || "0", // Use total_transactions if available
          token: "ETH", // Placeholder
          timestamp: latestResult.timestamp,
          status: "confirmed",
          endUserData: {
            address: event.address,
            user_profile_id: event.user_profile_id,
            user_category: event.user_category,
            user_category_name: event.user_category_name, // Use user_category_name instead of category
            end_user_likelihood: event.end_user_likelihood,
            confidence: event.confidence,
            is_anomaly: event.is_anomaly,
            cluster_id: event.cluster_id,
            behavior_patterns: event.behavior_patterns,
            suspicious_patterns: event.suspicious_patterns
          }
        })) || []
        
        setTransactions(transactions)
        setFilteredTransactions(transactions)
      } catch (err) {
        console.error("Error fetching transaction data:", err instanceof Error ? err.message : 'Failed to load transactions')
      } finally {
        setLoadingData(false)
      }
    }

    fetchTransactions()
    // Refresh every 30 seconds
    const interval = setInterval(fetchTransactions, 30000)
    return () => clearInterval(interval)
  }, [])
  
  // Filter transactions based on filterByEndUsers prop
  useEffect(() => {
    if (filterByEndUsers) {
      setFilteredTransactions(
        transactions.filter(tx => 
          tx.endUserData && 
          tx.endUserData.end_user_likelihood > 0.5 && 
          tx.endUserData.confidence > 0.5
        )
      )
    } else {
      setFilteredTransactions(transactions)
    }
  }, [transactions, filterByEndUsers])
  
  // If we're loading, show the skeleton
  if (isLoading || loadingData) {
    return <TableSkeleton />
  }

  const handleSort = (column: keyof Transaction | "endUserLikelihood") => {
    if (sort.column === column) {
      setSort({
        ...sort,
        direction: sort.direction === "asc" ? "desc" : "asc"
      })
    } else {
      setSort({
        column: column as keyof Transaction,
        direction: "asc"
      })
    }
  }

  const sortedTransactions = filteredTransactions.sort((a, b) => {
    if (!sort.column) return 0

    let valueA, valueB

    switch (sort.column) {
      case "id":
        valueA = a.id
        valueB = b.id
        break
      case "from":
        valueA = a.from
        valueB = b.from
        break
      case "to":
        valueA = a.to
        valueB = b.to
        break
      case "amount":
        valueA = parseFloat(a.amount)
        valueB = parseFloat(b.amount)
        break
      case "timestamp":
        // Convert timestamp strings to Date objects for proper chronological sorting
        valueA = new Date(a.timestamp).getTime()
        valueB = new Date(b.timestamp).getTime()
        break
      case "status":
        valueA = a.status
        valueB = b.status
        break
      case "endUserLikelihood":
        valueA = a.endUserData?.end_user_likelihood || 0
        valueB = b.endUserData?.end_user_likelihood || 0
        break
      default:
        return 0
    }

    if (valueA < valueB) {
      return sort.direction === "asc" ? -1 : 1
    }
    if (valueA > valueB) {
      return sort.direction === "asc" ? 1 : -1
    }
    return 0
  })

  const paginatedTransactions = showPagination
    ? sortedTransactions.slice((page - 1) * itemsPerPage, page * itemsPerPage)
    : sortedTransactions

  const totalPages = Math.ceil(sortedTransactions.length / itemsPerPage)

  return (
    <Card className="border-border/50">
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base font-medium">Recent Transactions</CardTitle>
        {filteredTransactions.length > 0 && !loadingData && (
          <div className="text-sm text-muted-foreground">
            {filterByEndUsers ? "End users only" : "All addresses"}
          </div>
        )}
      </CardHeader>
      <CardContent className="p-0">
        {loadingData || isLoading ? (
          <div className="p-4">
            <TableSkeleton />
          </div>
        ) : (
          <div className="w-full">
            <div className="rounded-md border min-h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-3 h-8"
                        onClick={() => handleSort("id")}
                      >
                        ID
                        <ArrowUpDown className="ml-2 h-3.5 w-3.5" />
                      </Button>
                    </TableHead>
                    <TableHead>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-3 h-8"
                        onClick={() => handleSort("from")}
                      >
                        From
                        <ArrowUpDown className="ml-2 h-3.5 w-3.5" />
                      </Button>
                    </TableHead>
                    <TableHead>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-3 h-8"
                        onClick={() => handleSort("to")}
                      >
                        To
                        <ArrowUpDown className="ml-2 h-3.5 w-3.5" />
                      </Button>
                    </TableHead>
                    <TableHead>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-3 h-8"
                        onClick={() => handleSort("amount")}
                      >
                        Amount
                        <ArrowUpDown className="ml-2 h-3.5 w-3.5" />
                      </Button>
                    </TableHead>
                    <TableHead>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-3 h-8"
                        onClick={() => handleSort("timestamp")}
                      >
                        Date
                        <ArrowUpDown className="ml-2 h-3.5 w-3.5" />
                      </Button>
                    </TableHead>
                    <TableHead className="text-center">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-3 h-8"
                        onClick={() => handleSort("status")}
                      >
                        Status
                        <ArrowUpDown className="ml-2 h-3.5 w-3.5" />
                      </Button>
                    </TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedTransactions.length > 0 ? (
                    paginatedTransactions.map((transaction) => (
                      <TransactionRow
                        key={transaction.id}
                        transaction={transaction}
                        expanded={expanded === transaction.id}
                        onExpandToggle={() =>
                          setExpanded(expanded === transaction.id ? null : transaction.id)
                        }
                      />
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={7} className="h-48 text-center">
                        No transactions found
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
            {showPagination && (
              <div className="flex items-center justify-between p-4">
                <div className="text-sm text-muted-foreground">
                  Showing <span className="font-medium">{((page - 1) * itemsPerPage) + 1}</span> to{" "}
                  <span className="font-medium">{Math.min(page * itemsPerPage, filteredTransactions.length)}</span> of{" "}
                  <span className="font-medium">{filteredTransactions.length}</span> transactions
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => Math.max(p - 1, 1))}
                    disabled={page === 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
                    disabled={page === totalPages}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

