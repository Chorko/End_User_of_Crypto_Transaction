"use client"

import { useEffect, useState } from "react"
import { ArrowLeftRight, Clock, Filter, Search, Wallet, ListFilter, Download } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TransactionGraph } from "./transaction-graph"
import { TransactionTable } from "./transaction-table"
import { TransactionStats } from "./transaction-stats"
import { AnalysisResults } from "./analysis-results"
import { Skeleton } from "@/components/ui/skeleton"
import { DropdownMenu, DropdownMenuContent, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuCheckboxItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Badge } from "@/components/ui/badge"

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [statsData, setStatsData] = useState({
    total_addresses: 0,
    total_end_users: 0,
    total_clusters: 0,
    category_distribution: {}
  })

  useEffect(() => {
    // Fetch analysis results
    const fetchData = async () => {
      try {
        // Fetch from API
        const response = await fetch('/api/results')
        if (!response.ok) {
          console.error("API error:", await response.text())
          setIsLoading(false)
          return
        }
        
        // Get all results (array)
        const allResults = await response.json()
        
        // Use the most recent result (last item in array)
        const data = allResults[allResults.length - 1]
        
        // Extract category distribution from visualization data if available
        let categoryDistribution = {}
        if (data.visualization_data?.nodes) {
          // Group nodes by their label/category
          categoryDistribution = data.visualization_data.nodes.reduce((acc: Record<string, number>, node: any) => {
            if (node.label) {
              acc[node.label] = (acc[node.label] || 0) + 1
            }
            return acc
          }, {})
        }
        
        // Calculate end users count - count users with end_user_likelihood above 0.5
        const endUsers = data.event_outputs?.filter((user: any) => 
          user.end_user_likelihood > 0.5
        ).length || 0
        
        // Calculate clusters count - exclude outlier cluster (-1)
        const clusters = data.clusters?.filter((cluster: any) => 
          cluster.id !== "-1"
        ).length || 0
        
        // Get documentation if available, for better category names
        const categoryNames = data.documentation?.user_categories || {}
        
        // Format category distribution for better display
        const formattedDistribution = Object.entries(categoryDistribution).reduce((acc, [key, value]) => {
          // Try to find a better name from documentation
          let displayName = key
          for (const [catId, info] of Object.entries(categoryNames)) {
            if (typeof info === 'object' && info.name === key) {
              displayName = info.name
              break
            }
          }
          acc[displayName] = value
          return acc
        }, {} as Record<string, number>)
        
        setStatsData({
          total_addresses: data.total_addresses || 0,
          total_end_users: endUsers,
          total_clusters: clusters,
          category_distribution: formattedDistribution
        })

        setIsLoading(false)
      } catch (err) {
        console.error("Error fetching data:", err)
        setIsLoading(false)
      }
    }
    
    fetchData()
  }, [])

  return (
    <div className="flex flex-col min-h-screen w-full max-w-full overflow-x-hidden">
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between max-w-full px-4 md:px-6">
          <div className="flex items-center gap-2 font-bold text-xl">
            <div className="size-8 rounded-full bg-gradient-to-br from-purple-600 to-cyan-400 flex items-center justify-center">
              <Wallet className="size-4 text-white" />
            </div>
            <span>CryptoTrack</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative w-64 hidden sm:block">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search transactions..." className="pl-8" />
            </div>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              <span className="hidden sm:inline">Filters</span>
            </Button>
          </div>
        </div>
      </header>
      <main className="flex-1 container py-6 w-full max-w-full px-4 md:px-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Clock className="mr-2 h-4 w-4" />
              <span className="hidden sm:inline">Last 24 hours</span>
            </Button>
            <Button variant="outline" size="sm">
              <ArrowLeftRight className="mr-2 h-4 w-4" />
              <span className="hidden sm:inline">Export</span>
            </Button>
          </div>
        </div>

        <div className="space-y-8 w-full">
          <Tabs defaultValue="overview" className="space-y-4 w-full">
            <div className="flex flex-col sm:flex-row sm:items-center gap-4">
              <TabsList className="h-auto flex-wrap">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="endusers">End Users</TabsTrigger>
                <TabsTrigger value="clusters">Clusters</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
              </TabsList>
              <div className="ml-auto flex items-center gap-2">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm" className="h-8 gap-1">
                      <ListFilter className="h-3.5 w-3.5" />
                      <span>Filter</span>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuLabel>Filter by</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuCheckboxItem checked>
                      Show end users only
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem>
                      Show high confidence ({'>'}0.8)
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem>
                      Show medium confidence (0.5-0.8)
                    </DropdownMenuCheckboxItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <Button size="sm" className="h-8 gap-1">
                  <Download className="h-3.5 w-3.5" />
                  <span>Export</span>
                </Button>
              </div>
            </div>
            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
                <TransactionStats
                  title="Total Addresses"
                  value={statsData.total_addresses.toLocaleString()}
                  change="+12.3%"
                  trend="up"
                  isLoading={isLoading}
                />
                <TransactionStats
                  title="End Users"
                  value={statsData.total_end_users.toLocaleString()}
                  change="+10.1%"
                  trend="up"
                  isLoading={isLoading}
                />
                <TransactionStats
                  title="Clusters"
                  value={statsData.total_clusters.toLocaleString()}
                  change="+5.4%"
                  trend="up"
                  isLoading={isLoading}
                />
                <TransactionStats
                  title="Categories"
                  value={Object.keys(statsData.category_distribution).length.toString()}
                  change="+2.1%"
                  trend="up"
                  isLoading={isLoading}
                />
              </div>
              
              <Card className="border-border/50">
                <CardHeader>
                  <CardTitle>Transaction Network</CardTitle>
                  <CardDescription>Visualization of cryptocurrency movement between addresses</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[400px] w-full">
                    <TransactionGraph />
                  </div>
                  <div className="mt-4 p-3 border border-purple-900/30 bg-purple-950/20 rounded-md text-xs text-muted-foreground m-4">
                    <p className="flex items-center">
                      <span className="font-semibold text-purple-400">Note:</span>
                      <span className="ml-2">
                        Graph shows transaction relationships between addresses. Larger nodes indicate higher end user likelihood.
                        Colors represent different user categories. Glowing nodes are identified end users.
                      </span>
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <TransactionTable showEndUserInfo={true} filterByEndUsers={false} />
            </TabsContent>
            <TabsContent value="endusers" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card className="col-span-2 min-h-[300px]">
                  <CardHeader>
                    <CardTitle>End User Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {isLoading ? (
                      <Skeleton className="h-[200px] w-full" />
                    ) : (
                      <div className="h-[200px] flex items-center justify-center bg-gradient-to-r from-indigo-900/20 to-purple-900/20 rounded-md overflow-hidden relative">
                        <div className="flex justify-between items-end w-full h-full px-8 pb-6 pt-2">
                          {Object.entries(statsData.category_distribution)
                            .filter(([category]) => 
                              // Only include end user categories (not exchanges)
                              !category.toLowerCase().includes('exchange') && 
                              !category.toLowerCase().includes('protocol')
                            )
                            .map(([category, count], index) => {
                              const colors = [
                                'bg-indigo-500', 
                                'bg-purple-500', 
                                'bg-cyan-500', 
                                'bg-teal-500'
                              ];
                              // Calculate height based on proportion
                              const maxHeight = 160; // Max bar height in pixels
                              const maxCount = Math.max(...Object.values(statsData.category_distribution) as number[]);
                              const height = maxCount > 0 ? Math.max(20, (count as number / maxCount) * maxHeight) : 20;
                              
                              return (
                                <div key={category} className="flex flex-col items-center">
                                  <div 
                                    className={`w-12 ${colors[index % colors.length]} rounded-t-md`}
                                    style={{ height: `${height}px` }}
                                  ></div>
                                  <div className="mt-2 text-xs">{category}</div>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
                <Card className="col-span-2 min-h-[300px]">
                  <CardHeader>
                    <CardTitle>End User Confidence</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {isLoading ? (
                      <Skeleton className="h-[200px] w-full" />
                    ) : (
                      <div className="h-[200px] flex items-center justify-center bg-gradient-to-r from-green-900/20 to-emerald-900/20 rounded-md overflow-hidden relative">
                        <div className="absolute inset-0 flex flex-col justify-center items-center">
                          <div className="text-4xl font-bold text-green-400">{Math.round((statsData.total_end_users / Math.max(statsData.total_addresses, 1)) * 100)}%</div>
                          <div className="text-sm text-green-300 mt-2">End user probability</div>
                        </div>
                        <div className="absolute bottom-0 w-full h-1/2 flex">
                          <div className="w-[40%] h-full bg-green-500/20 flex items-center justify-center">
                            <span className="text-xs">High</span>
                          </div>
                          <div className="w-[35%] h-full bg-yellow-500/20 flex items-center justify-center">
                            <span className="text-xs">Medium</span>
                          </div>
                          <div className="w-[25%] h-full bg-red-500/20 flex items-center justify-center">
                            <span className="text-xs">Low</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
              
              <Card className="border-border/50">
                <CardHeader>
                  <CardTitle>End User Behavior Patterns</CardTitle>
                  <CardDescription>Identified patterns across end user activities</CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-8 w-full" />
                      <Skeleton className="h-8 w-full" />
                      <Skeleton className="h-8 w-full" />
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1 mr-4">
                          <div className="text-sm font-medium mb-1">Regular Transactions</div>
                          <div className="h-2 w-full bg-muted overflow-hidden rounded-full">
                            <div className="h-full bg-indigo-500 w-[78%]"></div>
                          </div>
                        </div>
                        <div className="text-sm font-medium">78%</div>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex-1 mr-4">
                          <div className="text-sm font-medium mb-1">DeFi Interaction</div>
                          <div className="h-2 w-full bg-muted overflow-hidden rounded-full">
                            <div className="h-full bg-purple-500 w-[65%]"></div>
                          </div>
                        </div>
                        <div className="text-sm font-medium">65%</div>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex-1 mr-4">
                          <div className="text-sm font-medium mb-1">Gas Optimization</div>
                          <div className="h-2 w-full bg-muted overflow-hidden rounded-full">
                            <div className="h-full bg-blue-500 w-[85%]"></div>
                          </div>
                        </div>
                        <div className="text-sm font-medium">85%</div>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex-1 mr-4">
                          <div className="text-sm font-medium mb-1">Weekday Activity</div>
                          <div className="h-2 w-full bg-muted overflow-hidden rounded-full">
                            <div className="h-full bg-green-500 w-[89%]"></div>
                          </div>
                        </div>
                        <div className="text-sm font-medium">89%</div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
              
              <TransactionTable showEndUserInfo={true} filterByEndUsers={true} />
            </TabsContent>
            <TabsContent value="clusters" className="space-y-4">
              <AnalysisResults />
            </TabsContent>
            <TabsContent value="analytics" className="space-y-4">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Category Distribution</CardTitle>
                    <CardDescription>Distribution of addresses across different categories</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isLoading ? (
                      <Skeleton className="h-[300px] w-full" />
                    ) : (
                      <div className="h-[300px] relative">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-48 h-48 rounded-full border-8 border-background relative flex items-center justify-center">
                            <div className="absolute inset-0 border-8 border-r-indigo-500 border-transparent rounded-full rotate-45"></div>
                            <div className="absolute inset-0 border-8 border-t-purple-500 border-transparent rounded-full -rotate-15"></div>
                            <div className="absolute inset-0 border-8 border-l-teal-500 border-transparent rounded-full rotate-90"></div>
                            <div className="absolute inset-0 border-8 border-b-cyan-500 border-transparent rounded-full rotate-200"></div>
                            <div className="text-center text-sm">
                              <div className="font-bold">{statsData.total_addresses.toLocaleString()}</div>
                              <div className="text-xs text-muted-foreground">Total</div>
                            </div>
                          </div>
                        </div>
                        <div className="absolute bottom-4 w-full flex justify-around text-xs flex-wrap gap-2 px-2">
                          {Object.entries(statsData.category_distribution).map(([category, count], index) => {
                            const colors = ['bg-indigo-500', 'bg-purple-500', 'bg-teal-500', 'bg-cyan-500', 'bg-pink-500'];
                            const percentage = statsData.total_addresses > 0 
                              ? Math.round((count as number / statsData.total_addresses) * 100) 
                              : 0;
                              
                            return (
                              <div key={category} className="flex items-center">
                                <div className={`w-3 h-3 ${colors[index % colors.length]} mr-2 rounded-full`}></div>
                                <span>{category} ({percentage}%)</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Behavior Trends</CardTitle>
                    <CardDescription>Transactions and activity trends over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isLoading ? (
                      <Skeleton className="h-[300px] w-full" />
                    ) : (
                      <div className="h-[300px] relative p-4">
                        <div className="absolute bottom-10 left-0 right-0 h-[200px] flex items-end px-4">
                          {Array.from({ length: 7 }).map((_, i) => (
                            <div key={i} className="flex-1 flex flex-col items-center">
                              <div 
                                className="w-6 bg-gradient-to-t from-purple-600 to-blue-400 rounded-t-sm" 
                                style={{ 
                                  height: `${Math.floor(120 + Math.sin(i/2) * 60)}px`,
                                  opacity: i === 3 ? 1 : 0.7
                                }}
                              ></div>
                              <div className="mt-2 text-xs">{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i]}</div>
                            </div>
                          ))}
                        </div>
                        <div className="absolute top-4 left-4 rounded-md bg-background/80 backdrop-blur p-2 text-xs">
                          <div className="font-medium">Transaction Volume</div>
                          <div className="text-muted-foreground">Last 7 days</div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
                
                <Card className="md:col-span-2">
                  <CardHeader>
                    <CardTitle>Anomaly Detection</CardTitle>
                    <CardDescription>Unusual patterns and suspicious activities</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isLoading ? (
                      <div className="space-y-4">
                        <Skeleton className="h-12 w-full" />
                        <Skeleton className="h-12 w-full" />
                        <Skeleton className="h-12 w-full" />
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="p-3 border border-yellow-900/30 bg-yellow-950/20 rounded-md flex justify-between items-center">
                          <div>
                            <div className="text-sm font-medium">Unusual Transaction Pattern</div>
                            <div className="text-xs text-muted-foreground">Detected in 8 addresses</div>
                          </div>
                          <Badge variant="outline" className="border-yellow-500 text-yellow-500">Medium Risk</Badge>
                        </div>
                        <div className="p-3 border border-red-900/30 bg-red-950/20 rounded-md flex justify-between items-center">
                          <div>
                            <div className="text-sm font-medium">Rapid Exchange Transfers</div>
                            <div className="text-xs text-muted-foreground">Detected in 3 addresses</div>
                          </div>
                          <Badge variant="outline" className="border-red-500 text-red-500">High Risk</Badge>
                        </div>
                        <div className="p-3 border border-green-900/30 bg-green-950/20 rounded-md flex justify-between items-center">
                          <div>
                            <div className="text-sm font-medium">Unusual Gas Pricing</div>
                            <div className="text-xs text-muted-foreground">Detected in 12 addresses</div>
                          </div>
                          <Badge variant="outline" className="border-green-500 text-green-500">Low Risk</Badge>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}

