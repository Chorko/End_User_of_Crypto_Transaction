import { ArrowLeftRight, Clock, Filter, Search, Wallet } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TransactionGraph } from "./transaction-graph"
import { TransactionTable } from "./transaction-table"
import { TransactionStats } from "./transaction-stats"

export default function DashboardPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl">
            <div className="size-8 rounded-full bg-gradient-to-br from-purple-600 to-cyan-400 flex items-center justify-center">
              <Wallet className="size-4 text-white" />
            </div>
            <span>CryptoTrack</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative w-64">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search transactions..." className="pl-8" />
            </div>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filters
            </Button>
          </div>
        </div>
      </header>
      <main className="flex-1 container py-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Clock className="mr-2 h-4 w-4" />
              Last 24 hours
            </Button>
            <Button variant="outline" size="sm">
              <ArrowLeftRight className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-3 mb-6">
          <TransactionStats title="Total Transactions" value="2,543" change="+12.5%" trend="up" />
          <TransactionStats title="Total Volume" value="$1.2M" change="+8.2%" trend="up" />
          <TransactionStats title="Unique Addresses" value="487" change="-3.1%" trend="down" />
        </div>

        <Tabs defaultValue="overview" className="mb-6">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="transactions">Transactions</TabsTrigger>
            <TabsTrigger value="entities">Entities</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="space-y-6">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>Transaction Flow</CardTitle>
                <CardDescription>Visualization of cryptocurrency movement between addresses</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <TransactionGraph />
                </div>
                <div className="mt-4 p-3 border border-purple-900/30 bg-purple-950/20 rounded-md text-xs text-muted-foreground">
                  <p className="flex items-center">
                    <span className="font-semibold text-purple-400">Note:</span>
                    <span className="ml-2">
                      Entity clustering is currently under development. All visualizations and data shown are based on
                      synthetic data for demonstration purposes only.
                    </span>
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>Recent Transactions</CardTitle>
                <CardDescription>Latest transactions across the network</CardDescription>
              </CardHeader>
              <CardContent>
                <TransactionTable />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="transactions">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>All Transactions</CardTitle>
                <CardDescription>Comprehensive list of all tracked transactions</CardDescription>
              </CardHeader>
              <CardContent>
                <TransactionTable showPagination />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="entities">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>Identified Entities</CardTitle>
                <CardDescription>Clustered addresses linked to known entities</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <p className="mb-4">Entity visualization coming soon</p>
                  <div className="max-w-md mx-auto p-3 border border-purple-900/30 bg-purple-950/20 rounded-md text-xs text-left">
                    <p className="flex items-center">
                      <span className="font-semibold text-purple-400">Development Status:</span>
                      <span className="ml-2">
                        Entity clustering algorithms are currently under development. When complete, this module will
                        identify patterns and group related addresses.
                      </span>
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="reports">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>Analysis Reports</CardTitle>
                <CardDescription>Generated insights based on transaction patterns</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12 text-muted-foreground">Reports module coming soon</div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

