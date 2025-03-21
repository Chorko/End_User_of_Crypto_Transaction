import { ArrowDown, ArrowUp } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

interface TransactionStatsProps {
  title: string
  value: string
  change: string
  trend: "up" | "down"
  isLoading?: boolean
}

export function TransactionStats({ title, value, change, trend, isLoading = false }: TransactionStatsProps) {
  return (
    <Card className="border-border/50 h-[140px]">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-8 w-24" />
            <Skeleton className="h-4 w-32" />
          </div>
        ) : (
          <div className="min-h-[70px]">
            <div className="text-2xl font-bold">
              {value}
            </div>
            <p className="text-xs text-muted-foreground flex items-center mt-1">
              {trend === "up" ? (
                <ArrowUp className="mr-1 h-4 w-4 text-green-500" />
              ) : (
                <ArrowDown className="mr-1 h-4 w-4 text-red-500" />
              )}
              <span className={trend === "up" ? "text-green-500" : "text-red-500"}>{change}</span>
              <span className="ml-1">from last period</span>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

