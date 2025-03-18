"use client"

import { useState, useEffect } from "react"
import { ArrowUpDown, ChevronLeft, ChevronRight, ExternalLink, ChevronDown, ChevronUp } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"

import { Transaction, mockTransactions } from "./mock-data"

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
              <Badge variant="outline" className="mt-1 w-fit">
                {transaction.endUserData.user_category_name || "Unknown"}
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
            variant={
              transaction.status === "confirmed"
                ? "success"
                : transaction.status === "pending"
                ? "outline"
                : "destructive"
            }
            className="capitalize"
          >
            {transaction.status}
          </Badge>
        </TableCell>
        <TableCell className="text-right" onClick={onExpandToggle}>
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
  const [sort, setSort] = useState<{ column: keyof Transaction | null; direction: "asc" | "desc" }>({
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
    const fetchData = async () => {
      setLoadingData(true)
      try {
        // Fetch transactions
        const transactionResponse = await fetch('/api/transactions')
        let transactionData = []
        
        if (transactionResponse.ok) {
          transactionData = await transactionResponse.json()
        } else {
          // If API fails, use mock data
          transactionData = mockTransactions
        }
        
        // If we need end user information, fetch and integrate it
        if (showEndUserInfo) {
          try {
            const endUserResponse = await fetch('/api/results')
            if (endUserResponse.ok) {
              const endUserData = await endUserResponse.json()
              
              // Combine data
              if (endUserData.event_outputs) {
                transactionData = transactionData.map((tx: Transaction) => {
                  const endUser = endUserData.event_outputs.find(
                    (user: any) => user.address.toLowerCase() === tx.from.toLowerCase()
                  )
                  return {
                    ...tx,
                    endUserData: endUser ? {
                      ...endUser,
                      user_category_name: endUser.category || "Unknown"
                    } : undefined
                  }
                })
              }
            }
          } catch (error) {
            console.error("Error fetching end user data:", error)
          }
        }
        
        setTransactions(transactionData)
      } catch (error) {
        console.error("Error fetching transaction data:", error)
        // Use mock data if API fails
        setTransactions(mockTransactions)
      } finally {
        // Simulate a loading delay
        setTimeout(() => {
          setLoadingData(false)
        }, 1000)
      }
    }
    
    fetchData()
  }, [showEndUserInfo])
  
  // Filter transactions based on filterByEndUsers prop
  useEffect(() => {
    if (filterByEndUsers) {
      setFilteredTransactions(
        transactions.filter(tx => 
          tx.endUserData && tx.endUserData.end_user_likelihood > 0.5
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

  const handleSort = (column: string) => {
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
        valueA = a.timestamp
        valueB = b.timestamp
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
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[150px]">
                <div className="flex items-center cursor-pointer" onClick={() => handleSort("id")}>
                  Transaction
                  {sort.column === "id" && (
                    <ArrowUpDown className={`ml-2 h-4 w-4 ${sort.direction === "asc" ? "rotate-180" : ""}`} />
                  )}
                </div>
              </TableHead>
              <TableHead>
                <div className="flex items-center cursor-pointer" onClick={() => handleSort("from")}>
                  From
                  {sort.column === "from" && (
                    <ArrowUpDown className={`ml-2 h-4 w-4 ${sort.direction === "asc" ? "rotate-180" : ""}`} />
                  )}
                </div>
              </TableHead>
              <TableHead>
                <div className="flex items-center cursor-pointer" onClick={() => handleSort("to")}>
                  To
                  {sort.column === "to" && (
                    <ArrowUpDown className={`ml-2 h-4 w-4 ${sort.direction === "asc" ? "rotate-180" : ""}`} />
                  )}
                </div>
              </TableHead>
              <TableHead>
                <div className="flex items-center cursor-pointer" onClick={() => handleSort("amount")}>
                  Amount
                  {sort.column === "amount" && (
                    <ArrowUpDown className={`ml-2 h-4 w-4 ${sort.direction === "asc" ? "rotate-180" : ""}`} />
                  )}
                </div>
              </TableHead>
              <TableHead className="hidden lg:table-cell">
                <div className="flex items-center cursor-pointer" onClick={() => handleSort("timestamp")}>
                  Timestamp
                  {sort.column === "timestamp" && (
                    <ArrowUpDown className={`ml-2 h-4 w-4 ${sort.direction === "asc" ? "rotate-180" : ""}`} />
                  )}
                </div>
              </TableHead>
              <TableHead className="hidden md:table-cell">
                <div className="flex items-center justify-center cursor-pointer" onClick={() => handleSort("status")}>
                  Status
                  {sort.column === "status" && (
                    <ArrowUpDown className={`ml-2 h-4 w-4 ${sort.direction === "asc" ? "rotate-180" : ""}`} />
                  )}
                </div>
              </TableHead>
              <TableHead className="w-[50px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loadingData
              ? Array.from({ length: 5 }).map((_, i) => (
                  <TransactionRow key={i} transaction={{} as Transaction} isLoading={true} />
                ))
              : paginatedTransactions.map((transaction) => (
                  <TransactionRow 
                    key={transaction.id} 
                    transaction={transaction} 
                    expanded={transaction.id === expanded}
                    onExpandToggle={() => setExpanded(transaction.id === expanded ? null : transaction.id)}
                  />
                ))}
            {!loadingData && paginatedTransactions.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} className="h-24 text-center">No transactions found.</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {showPagination && totalPages > 1 && (
        <div className="flex items-center justify-end gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1 || loadingData}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <div className="text-xs text-muted-foreground">
            Page {page} of {totalPages}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages || loadingData}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}

