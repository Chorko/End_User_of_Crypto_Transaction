"use client"

import { useEffect, useRef, useState } from "react"
import * as d3 from "d3"
import { motion } from "framer-motion"
import { ZoomIn, ZoomOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"

// Define our data types
interface Node {
  id: string
  group: number
  value: number
  label?: string
  is_end_user?: boolean
  cluster_id?: number
  likelihood?: number
}

interface Link {
  source: string
  target: string
  value: number
}

interface GraphData {
  nodes: Node[]
  links: Link[]
}

// Map user categories to colors
const NODE_COLORS = {
  0: "rgba(99, 102, 241, 0.8)", // Indigo - Individual/Retail User
  1: "rgba(236, 72, 153, 0.8)", // Pink - Institutional/Large Investor
  2: "rgba(139, 92, 246, 0.8)", // Purple - Exchange/Protocol Account
  3: "rgba(45, 212, 191, 0.8)", // Teal - DeFi User
  4: "rgba(34, 211, 238, 0.8)", // Cyan - NFT Trader/Collector
  "default": "rgba(156, 163, 175, 0.8)", // Gray - Unknown/Other
}

// Map user categories to labels
const NODE_LABELS = {
  0: "Individual/Retail User",
  1: "Institutional/Large Investor",
  2: "Exchange/Protocol Account",
  3: "DeFi User",
  4: "NFT Trader/Collector",
  "default": "Unknown"
}

export function TransactionGraph() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [isLoading, setIsLoading] = useState(true)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  
  // Fetch data from API
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch analysis results
        const response = await fetch('/api/results')
        if (!response.ok) {
          throw new Error(`Error fetching data: ${response.statusText}`)
        }
        
        const allResults = await response.json()
        
        // Use the most recent result (last item in array)
        const latestResult = allResults[allResults.length - 1]
        
        // Generate graph data from the results
        const graphData = generateGraphData(latestResult)
        setGraphData(graphData)
        setIsLoading(false)
      } catch (error) {
        console.error("Error fetching data:", error)
        setIsLoading(false)
      }
    }
    
    fetchData()
  }, [])
  
  // Generate the graph data from the API response
  const generateGraphData = (results: any): GraphData => {
    if (!results || !results.visualization_data) {
      return { nodes: [], links: [] }
    }

    const { visualization_data } = results

    // Use the visualization_data directly if it exists
    if (visualization_data.nodes && visualization_data.links) {
      try {
        // If there are too many nodes, select a sample of the most significant ones
        let formattedNodes = []
        
        // For large datasets, prioritize by value (transaction count)
        // but limit to a reasonable number for performance
        const maxNodes = 120 // Increase from default (which was likely 50)
        let nodesToUse = visualization_data.nodes
        
        if (visualization_data.nodes.length > maxNodes) {
          // Sort by value (typically transaction count) and take the top ones
          nodesToUse = [...visualization_data.nodes]
            .sort((a, b) => (b.value || 1) - (a.value || 1))
            .slice(0, maxNodes)
          
          console.log(`Limited visualization from ${visualization_data.nodes.length} to ${maxNodes} nodes for performance`)
        }
        
        // Ensure nodes have the proper format
        formattedNodes = nodesToUse.map((node: any) => ({
          id: node.id || `node-${Math.random().toString(36).slice(2,8)}`,
          group: typeof node.group === 'number' ? node.group : 0,
          value: typeof node.value === 'number' ? node.value : 1,
          label: node.label || "Unknown"
        }))

        // Ensure links have the proper format and reference existing nodes
        const nodeIds = new Set(formattedNodes.map(node => node.id))
        const formattedLinks = visualization_data.links
          .filter((link: any) => 
            link.source && 
            link.target && 
            nodeIds.has(typeof link.source === 'object' ? link.source.id : link.source) && 
            nodeIds.has(typeof link.target === 'object' ? link.target.id : link.target)
          )
          .map((link: any) => ({
            source: typeof link.source === 'object' ? link.source.id : link.source,
            target: typeof link.target === 'object' ? link.target.id : link.target,
            value: typeof link.value === 'number' ? link.value : 1
          }))

        // If we don't have enough links, generate additional ones based on cluster memberships
        if (formattedLinks.length < formattedNodes.length / 2) {
          console.log("Not enough links in visualization data, enhancing connectivity...")
          
          // Group nodes by their cluster/group
          const nodesByGroup = formattedNodes.reduce((groups: {[key: string]: Node[]}, node) => {
            const groupId = String(node.group || 0)
            if (!groups[groupId]) groups[groupId] = []
            groups[groupId].push(node)
            return groups
          }, {})
          
          // For each group with multiple nodes, create links between them
          Object.values(nodesByGroup).forEach(groupNodes => {
            if (groupNodes.length <= 1) return
            
            // Create a "star" topology within each group
            // Pick a central node (highest value)
            const centralNode = [...groupNodes].sort((a, b) => (b.value || 0) - (a.value || 0))[0]
            
            // Connect other nodes to the central node
            groupNodes
              .filter(node => node.id !== centralNode.id)
              .forEach(node => {
                formattedLinks.push({
                  source: centralNode.id,
                  target: node.id,
                  value: 2
                })
                
                // Add some additional connections for larger groups
                if (groupNodes.length >= 5 && Math.random() > 0.7) {
                  // Connect to another random node in the same group
                  const otherNode = groupNodes[Math.floor(Math.random() * groupNodes.length)]
                  if (otherNode.id !== node.id && otherNode.id !== centralNode.id) {
                    formattedLinks.push({
                      source: node.id,
                      target: otherNode.id,
                      value: 1
                    })
                  }
                }
              })
          })
        }

        return {
          nodes: formattedNodes,
          links: formattedLinks
        }
      } catch (error) {
        console.error("Error formatting visualization data:", error)
        // Fall through to the cluster-based visualization fallback
      }
    }

    // Fallback to a simple graph derived from clusters if visualization_data is not properly formatted
    if (results.clusters) {
      const nodes: Node[] = []
      const links: Link[] = []
      
      // Create a central hub node for each cluster
      results.clusters.forEach((cluster: any) => {
        if (cluster.id !== "-1") { // Skip outlier cluster
          // Add central node for the cluster
          nodes.push({
            id: `cluster-${cluster.id}`,
            group: 0,
            value: cluster.size || 1,
            label: `Cluster ${cluster.id}`
          })
          
          // Add nodes for each address
          cluster.addresses.forEach((address: string) => {
            nodes.push({
              id: address,
              group: parseInt(cluster.id) || 1,
              value: 1,
              label: "Address"
            })
            
            // Link to cluster center
          links.push({
              source: address,
              target: `cluster-${cluster.id}`,
              value: 1
            })
          })
      }
    })
    
    return { nodes, links }
    }
    
    return { nodes: [], links: [] }
  }

  // Render graph once data is available
  useEffect(() => {
    if (!svgRef.current || !graphData || isLoading) return

    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight

    // Clear any existing SVG content
    d3.select(svgRef.current).selectAll("*").remove()

    const data = graphData

    // Create the force simulation
    const simulation = d3
      .forceSimulation(data.nodes as d3.SimulationNodeDatum[])
      .force(
        "link",
        d3
          .forceLink(data.links)
          .id((d: any) => d.id)
          .distance(70),
      )
      .force("charge", d3.forceManyBody().strength(-150))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(width / 2).strength(0.1))
      .force("y", d3.forceY(height / 2).strength(0.1))
      .force("collision", d3.forceCollide().radius((d: any) => Math.sqrt(d.value) * 2 + 5))

    // Create a group for the graph
    const svg = d3.select(svgRef.current)
    const g = svg.append("g")

    // Define gradient for links
    const defs = g.append("defs")
    
    // Create gradient for each category
    Object.entries(NODE_COLORS).forEach(([category, color]) => {
      const gradient = defs
        .append("linearGradient")
        .attr("id", `link-gradient-${category}`)
        .attr("gradientUnits", "userSpaceOnUse")
      
      gradient.append("stop").attr("offset", "0%").attr("stop-color", color)
      gradient.append("stop").attr("offset", "100%").attr("stop-color", "rgba(156, 163, 175, 0.6)")
    })
    
    // Create gold gradient for end users
    const goldGradient = defs
      .append("radialGradient")
      .attr("id", "gold-gradient")
      .attr("cx", "50%")
      .attr("cy", "50%")
      .attr("r", "50%")
      .attr("fx", "50%")
      .attr("fy", "50%")
    
    goldGradient.append("stop").attr("offset", "0%").attr("stop-color", "#fbbf24") // Amber-400
    goldGradient.append("stop").attr("offset", "70%").attr("stop-color", "#f59e0b") // Amber-500
    goldGradient.append("stop").attr("offset", "100%").attr("stop-color", "#d97706") // Amber-600
    
    // Define a glow filter for end users
    const filter = defs.append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    
    // Add gold glow effect
    const feFlood = filter.append("feFlood")
      .attr("flood-color", "#f59e0b")  // Amber-500
      .attr("result", "flood");
      
    const feComposite = filter.append("feComposite")
      .attr("in", "flood")
      .attr("in2", "SourceGraphic")
      .attr("operator", "in")
      .attr("result", "color");
      
    filter.append("feGaussianBlur")
      .attr("in", "color")
      .attr("stdDeviation", "3")
      .attr("result", "coloredBlur");
    
    const femerge = filter.append("feMerge");
    femerge.append("feMergeNode").attr("in", "coloredBlur");
    femerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Create links
    const link = g
      .append("g")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("stroke", (d: any) => {
        const sourceNode = data.nodes.find(n => n.id === d.source.id || n.id === d.source)
        const category = sourceNode?.group || "default"
        return `url(#link-gradient-${category})`
      })
      .attr("stroke-width", (d) => Math.sqrt(d.value) * 1.5)

    // Create nodes group
    const node = g
      .append("g")
      .selectAll("g")
      .data(data.nodes)
      .join("g")
      .call(drag(simulation) as any)

    // Add circles to nodes
    node
      .append("circle")
      .attr("r", (d) => Math.sqrt(d.value) * 2)
      .attr("fill", (d) => {
        // Use gold color for end users, otherwise use category color
        if (d.is_end_user) {
          return "url(#gold-gradient)"
        }
        return NODE_COLORS[d.group as keyof typeof NODE_COLORS] || NODE_COLORS.default
      })
      .attr("stroke", (d) => d.is_end_user ? "#f59e0b" : "#10101a")
      .attr("stroke-width", (d) => d.is_end_user ? 2 : 1.5)
      .attr("filter", (d) => d.is_end_user ? "url(#glow)" : "")

    // Add labels to nodes
    node
      .append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", "10px")
      .text((d) => d.id.slice(0, 6) + "...")

    // Add tooltips
    node.append("title").text((d) => {
      let tooltip = `Address: ${d.id}\nCategory: ${d.label}`
      if (d.likelihood !== undefined) {
        tooltip += `\nEnd User Likelihood: ${(d.likelihood * 100).toFixed(0)}%`
      }
      if (d.cluster_id !== undefined) {
        tooltip += `\nCluster: ${d.cluster_id}`
      }
      return tooltip
    })

    // Update positions on each tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y)

      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`)
    })

    // Add zoom behavior
    const zoomBehavior = d3
      .zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform)
        setZoom(event.transform.k)
      })

    svg.call(zoomBehavior as any)

    // Drag functionality
    function drag(simulation: d3.Simulation<d3.SimulationNodeDatum, undefined>) {
      function dragstarted(event: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        event.subject.fx = event.subject.x
        event.subject.fy = event.subject.y
      }

      function dragged(event: any) {
        event.subject.fx = event.x
        event.subject.fy = event.y
      }

      function dragended(event: any) {
        if (!event.active) simulation.alphaTarget(0)
        event.subject.fx = null
        event.subject.fy = null
      }

      return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended)
    }

    // Handle resize
    const handleResize = () => {
      if (!svgRef.current) return

      const width = svgRef.current.clientWidth
      const height = svgRef.current.clientHeight

      simulation.force("center", d3.forceCenter(width / 2, height / 2))
      simulation.force("x", d3.forceX(width / 2).strength(0.1))
      simulation.force("y", d3.forceY(height / 2).strength(0.1))
      simulation.alpha(0.3).restart()
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      simulation.stop()
    }
  }, [graphData, isLoading])

  const handleZoomIn = () => {
    if (!svgRef.current) return
    d3.select(svgRef.current).transition().duration(300).call(d3.zoom().scaleTo as any, zoom * 1.2)
  }

  const handleZoomOut = () => {
    if (!svgRef.current) return
    d3.select(svgRef.current).transition().duration(300).call(d3.zoom().scaleTo as any, zoom / 1.2)
  }

  return (
    <div className="relative w-full h-full">
      {isLoading ? (
        <div className="w-full h-full flex items-center justify-center">
          <Skeleton className="w-full h-full absolute" />
        </div>
      ) : (
        <div className="w-full h-full overflow-hidden relative">
          <div className="absolute top-4 right-4 flex space-x-2 z-10">
            <Button 
              variant="outline" 
              size="icon" 
              className="h-8 w-8 bg-background/80 backdrop-blur" 
              onClick={handleZoomIn}
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button 
              variant="outline" 
              size="icon" 
              className="h-8 w-8 bg-background/80 backdrop-blur" 
              onClick={handleZoomOut}
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
          </div>
          <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm p-3 rounded-md text-xs z-10">
            <div className="font-medium mb-2">Categories</div>
            <div className="space-y-1.5">
              {Object.entries(NODE_LABELS).map(([key, label]) => {
                if (key === "default") return null
                return (
                  <div key={key} className="flex items-center">
                    <div 
                      className="w-3 h-3 rounded-full mr-2" 
                      style={{ backgroundColor: NODE_COLORS[key as keyof typeof NODE_COLORS] }}
                    ></div>
                    <span>{label}</span>
                  </div>
                )
              })}
            </div>
            <div className="mt-3 pt-2 border-t border-muted">
              <div className="flex items-center">
                <div className="w-4 h-4 rounded-full mr-2 relative">
                  <div className="absolute inset-0 rounded-full bg-amber-400"></div>
                  <div className="absolute inset-0 rounded-full bg-yellow-300/30 animate-ping"></div>
                </div>
                <span>End User (high likelihood)</span>
              </div>
            </div>
          </div>
          <svg 
            ref={svgRef} 
            className="w-full h-full" 
            style={{ minHeight: "400px", maxWidth: "100%" }}
            viewBox="0 0 800 400"
            preserveAspectRatio="xMidYMid meet"
          ></svg>
        </div>
      )}
    </div>
  )
}

