import type React from "react"
import Link from "next/link"
import { ArrowRight, BarChart3, Database, Globe, Search } from "lucide-react"

import { Button } from "@/components/ui/button"
import { HeroAnimation } from "./hero-animation"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl">
            <div className="size-8 rounded-full bg-gradient-to-br from-purple-600 to-cyan-400 flex items-center justify-center">
              <Database className="size-4 text-white" />
            </div>
            <span>CryptoTrack</span>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <Link href="#features" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Features
            </Link>
            <Link href="#explorer" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Explorer
            </Link>
            <Link href="#insights" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Insights
            </Link>
            <Link href="#about" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              About
            </Link>
          </nav>
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/login">Login</Link>
            </Button>
            <Button
              size="sm"
              className="bg-gradient-to-r from-purple-600 to-cyan-500 hover:from-purple-700 hover:to-cyan-600"
            >
              <Link href="/dashboard">Get Started</Link>
            </Button>
          </div>
        </div>
      </header>
      <main className="flex-1">
        <section className="container py-24 space-y-8 md:py-32">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
            <h1 className="text-4xl font-bold leading-tight tracking-tighter md:text-6xl lg:leading-[1.1]">
              Unmasking Cryptocurrency{" "}
              <span className="bg-gradient-to-r from-purple-400 via-cyan-400 to-fuchsia-500 bg-clip-text text-transparent">
                Transactions
              </span>
            </h1>
            <p className="max-w-[46rem] text-lg text-muted-foreground sm:text-xl">
              Track, analyze, and visualize blockchain transactions with our powerful and intuitive platform. Gain
              insights into cryptocurrency flows and entity relationships.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button
                asChild
                className="bg-gradient-to-r from-purple-600 to-cyan-500 hover:from-purple-700 hover:to-cyan-600"
              >
                <Link href="/dashboard">
                  Explore Dashboard
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button variant="outline">
                <Link href="/docs">View Documentation</Link>
              </Button>
            </div>
          </div>
          <div className="mx-auto max-w-5xl">
            <HeroAnimation />
          </div>
        </section>

        <section id="features" className="container py-20 space-y-16">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-4 text-center">
            <h2 className="text-3xl font-bold leading-tight tracking-tighter md:text-4xl">Powerful Features</h2>
            <p className="max-w-[46rem] text-muted-foreground sm:text-lg">
              Our platform provides comprehensive tools to analyze cryptocurrency transactions and identify patterns.
            </p>
          </div>

          <div className="mx-auto grid justify-center gap-8 sm:grid-cols-2 md:grid-cols-3">
            <FeatureCard
              icon={<BarChart3 className="h-10 w-10 text-purple-500" />}
              title="Interactive Dashboard"
              description="Visualize transaction flows with interactive charts and real-time data updates."
            />
            <FeatureCard
              icon={<Search className="h-10 w-10 text-cyan-500" />}
              title="Transaction Explorer"
              description="Search and analyze transaction histories with detailed breakdowns and entity information."
            />
            <FeatureCard
              icon={<Globe className="h-10 w-10 text-fuchsia-500" />}
              title="Entity Identification"
              description="Identify and cluster related addresses using advanced machine learning algorithms."
            />
          </div>
        </section>
      </main>
      <footer className="border-t border-border/40 bg-background/95">
        <div className="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0">
          <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
            <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
              &copy; {new Date().getFullYear()} CryptoTrack. All rights reserved.
            </p>
          </div>
          <div className="flex gap-4">
            <Link href="#" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Terms
            </Link>
            <Link href="#" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Privacy
            </Link>
            <Link href="#" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="group relative overflow-hidden rounded-lg border border-border/50 bg-background/50 p-6 shadow-sm transition-all hover:border-border hover:shadow-md">
      <div className="mb-4">{icon}</div>
      <h3 className="mb-2 text-xl font-bold">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-purple-900/10 to-cyan-900/10 opacity-0 transition-opacity group-hover:opacity-100" />
    </div>
  )
}

