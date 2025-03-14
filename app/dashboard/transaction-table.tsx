"use client"

import { useState } from "react"
import { ArrowUpDown, ChevronLeft, ChevronRight, ExternalLink } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface Transaction {
  id: string
  from: string
  to: string
  amount: string
  token: string
  timestamp: string
  status: "confirmed" | "pending" | "failed"
}

const transactions: Transaction[] = [
  {
    id: "0x1a2b3c4d5e6f",
    from: "0x7g8h9i0j1k2l",
    to: "0x3m4n5o6p7q8r",
    amount: "1.245",
    token: "ETH",
    timestamp: "2023-03-15 14:30:45",
    status: "confirmed",
  },
  {
    id: "0x2b3c4d5e6f7g",
    from: "0x8h9i0j1k2l3m",
    to: "0x4n5o6p7q8r9s",
    amount: "425.75",
    token: "USDT",
    timestamp: "2023-03-15 13:25:12",
    status: "confirmed",
  },
  {
    id: "0x3c4d5e6f7g8h",
    from: "0x9i0j1k2l3m4n",
    to: "0x5o6p7q8r9s0t",
    amount: "0.078",
    token: "BTC",
    timestamp: "2023-03-15 12:18:33",
    status: "pending",
  },
  {
    id: "0x4d5e6f7g8h9i",
    from: "0x0j1k2l3m4n5o",
    to: "0x6p7q8r9s0t1u",
    amount: "1,250.00",
    token: "USDC",
    timestamp: "2023-03-15 11:05:27",
    status: "confirmed",
  },
  {
    id: "0x5e6f7g8h9i0j",
    from: "0x1k2l3m4n5o6p",
    to: "0x7q8r9s0t1u2v",
    amount: "15.5",
    token: "SOL",
    timestamp: "2023-03-15 10:42:19",
    status: "failed",
  },
]

export function TransactionTable({ showPagination = false }: { showPagination?: boolean }) {
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc")

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
  }

  const truncateAddress = (address: string) => {
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "confirmed":
        return "bg-green-500/20 text-green-500 hover:bg-green-500/30"
      case "pending":
        return "bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30"
      case "failed":
        return "bg-red-500/20 text-red-500 hover:bg-red-500/30"
      default:
        return "bg-gray-500/20 text-gray-500 hover:bg-gray-500/30"
    }
  }

  return (
    <div>
      <div className="rounded-md border border-border/50">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[100px]">
                <Button variant="ghost" size="sm" onClick={() => handleSort("id")}>
                  TX Hash
                  <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
              </TableHead>
              <TableHead>
                <Button variant="ghost" size="sm" onClick={() => handleSort("from")}>
                  From
                  <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
              </TableHead>
              <TableHead>
                <Button variant="ghost" size="sm" onClick={() => handleSort("to")}>
                  To
                  <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
              </TableHead>
              <TableHead className="text-right">
                <Button variant="ghost" size="sm" onClick={() => handleSort("amount")}>
                  Amount
                  <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
              </TableHead>
              <TableHead>
                <Button variant="ghost" size="sm" onClick={() => handleSort("timestamp")}>
                  Time
                  <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
              </TableHead>
              <TableHead>
                <Button variant="ghost" size="sm" onClick={() => handleSort("status")}>
                  Status
                  <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {transactions.map((transaction) => (
              <TableRow key={transaction.id}>
                <TableCell className="font-mono">
                  <div className="flex items-center">
                    {truncateAddress(transaction.id)}
                    <Button variant="ghost" size="icon" className="h-6 w-6 ml-1">
                      <ExternalLink className="h-3 w-3" />
                      <span className="sr-only">View transaction</span>
                    </Button>
                  </div>
                </TableCell>
                <TableCell className="font-mono">{truncateAddress(transaction.from)}</TableCell>
                <TableCell className="font-mono">{truncateAddress(transaction.to)}</TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-2">
                    <span>{transaction.amount}</span>
                    <span className="text-xs text-muted-foreground">{transaction.token}</span>
                  </div>
                </TableCell>
                <TableCell>{transaction.timestamp}</TableCell>
                <TableCell>
                  <Badge variant="outline" className={getStatusColor(transaction.status)}>
                    {transaction.status}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="mt-2 text-xs text-muted-foreground italic">
        * Transaction data shown is synthetic and for demonstration purposes only
      </div>

      {showPagination && (
        <div className="flex items-center justify-end space-x-2 py-4">
          <Button variant="outline" size="sm">
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>
          <Button variant="outline" size="sm">
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}

