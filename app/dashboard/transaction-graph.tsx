"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"
import { motion } from "framer-motion"

// Define our data types
interface Node {
  id: string
  group: number
  value: number
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

export function TransactionGraph() {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    // Sample data - in a real app, this would come from an API
    const data: GraphData = {
      nodes: [
        { id: "0x1a2b3c", group: 1, value: 20 },
        { id: "0x4d5e6f", group: 1, value: 15 },
        { id: "0x7g8h9i", group: 2, value: 25 },
        { id: "0xjklmno", group: 2, value: 18 },
        { id: "0xpqrstu", group: 3, value: 30 },
        { id: "0xvwxyz1", group: 3, value: 22 },
        { id: "0x234567", group: 1, value: 12 },
        { id: "0x890abc", group: 2, value: 28 },
        { id: "0xdef123", group: 3, value: 16 },
        { id: "0x456789", group: 1, value: 24 },
      ],
      links: [
        { source: "0x1a2b3c", target: "0x7g8h9i", value: 5 },
        { source: "0x1a2b3c", target: "0xjklmno", value: 3 },
        { source: "0x4d5e6f", target: "0xpqrstu", value: 8 },
        { source: "0x7g8h9i", target: "0xvwxyz1", value: 6 },
        { source: "0xjklmno", target: "0x234567", value: 4 },
        { source: "0xpqrstu", target: "0x890abc", value: 7 },
        { source: "0xvwxyz1", target: "0xdef123", value: 2 },
        { source: "0x234567", target: "0x456789", value: 9 },
      ],
    }

    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight

    // Clear any existing SVG content
    d3.select(svgRef.current).selectAll("*").remove()

    // Create the force simulation
    const simulation = d3
      .forceSimulation(data.nodes as d3.SimulationNodeDatum[])
      .force(
        "link",
        d3
          .forceLink(data.links)
          .id((d: any) => d.id)
          .distance(100),
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(width / 2).strength(0.1))
      .force("y", d3.forceY(height / 2).strength(0.1))

    // Create a group for the graph
    const svg = d3.select(svgRef.current)

    // Define gradient for links
    const defs = svg.append("defs")

    const gradient = defs.append("linearGradient").attr("id", "link-gradient").attr("gradientUnits", "userSpaceOnUse")

    gradient.append("stop").attr("offset", "0%").attr("stop-color", "rgba(139, 92, 246, 0.7)")

    gradient.append("stop").attr("offset", "100%").attr("stop-color", "rgba(34, 211, 238, 0.7)")

    // Create links
    const link = svg
      .append("g")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("stroke", "url(#link-gradient)")
      .attr("stroke-width", (d) => Math.sqrt(d.value))

    // Create nodes
    const node = svg
      .append("g")
      .selectAll("circle")
      .data(data.nodes)
      .join("circle")
      .attr("r", (d) => Math.sqrt(d.value) * 2)
      .attr("fill", (d) => {
        const colors = ["rgba(139, 92, 246, 0.8)", "rgba(34, 211, 238, 0.8)", "rgba(232, 121, 249, 0.8)"]
        return colors[d.group - 1]
      })
      .attr("stroke", "#10101a")
      .attr("stroke-width", 1.5)
      .call(drag(simulation) as any)

    // Add tooltips
    node.append("title").text((d) => d.id)

    // Update positions on each tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y)

      node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y)
    })

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
  }, [])

  return (
    <motion.div
      className="w-full h-full"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <svg ref={svgRef} className="w-full h-full bg-black/20 rounded-lg" />
      <div className="mt-2 text-xs text-muted-foreground italic">
        * Visualization based on synthetic blockchain data
      </div>
    </motion.div>
  )
}

