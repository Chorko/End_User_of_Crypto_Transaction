"use client"

import { useEffect, useRef } from "react"
import { motion } from "framer-motion"

export function HeroAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const setCanvasDimensions = () => {
      canvas.width = canvas.clientWidth
      canvas.height = canvas.clientHeight
    }

    setCanvasDimensions()
    window.addEventListener("resize", setCanvasDimensions)

    // Node class for blockchain visualization
    class Node {
      x: number
      y: number
      radius: number
      color: string
      connections: Node[]
      vx: number
      vy: number

      constructor(x: number, y: number, radius: number, color: string) {
        this.x = x
        this.y = y
        this.radius = radius
        this.color = color
        this.connections = []
        this.vx = (Math.random() - 0.5) * 0.5
        this.vy = (Math.random() - 0.5) * 0.5
      }

      draw() {
        if (!ctx) return

        ctx.beginPath()
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
        ctx.fillStyle = this.color
        ctx.fill()

        // Draw connections
        this.connections.forEach((node) => {
          ctx.beginPath()
          ctx.moveTo(this.x, this.y)
          ctx.lineTo(node.x, node.y)
          ctx.strokeStyle = "rgba(139, 92, 246, 0.2)"
          ctx.lineWidth = 1
          ctx.stroke()
        })
      }

      update() {
        // Simple boundary checking
        if (!canvas) return;
        
        if (this.x + this.radius > canvas.width || this.x - this.radius < 0) {
          this.vx = -this.vx
        }

        if (this.y + this.radius > canvas.height || this.y - this.radius < 0) {
          this.vy = -this.vy
        }

        this.x += this.vx
        this.y += this.vy
      }
    }

    // Create nodes
    const nodes: Node[] = []
    const nodeCount = 20

    for (let i = 0; i < nodeCount; i++) {
      const radius = Math.random() * 3 + 2
      if (!canvas) return;
      const x = Math.random() * (canvas.width - radius * 2) + radius
      const y = Math.random() * (canvas.height - radius * 2) + radius

      // Generate a color from our palette
      const colors = [
        "rgba(139, 92, 246, 0.8)", // purple
        "rgba(34, 211, 238, 0.8)", // cyan
        "rgba(232, 121, 249, 0.8)", // fuchsia
      ]
      const color = colors[Math.floor(Math.random() * colors.length)]

      nodes.push(new Node(x, y, radius, color))
    }

    // Create connections between nodes
    nodes.forEach((node) => {
      const connectionCount = Math.floor(Math.random() * 3) + 1
      for (let i = 0; i < connectionCount; i++) {
        const randomNode = nodes[Math.floor(Math.random() * nodes.length)]
        if (randomNode !== node && !node.connections.includes(randomNode)) {
          node.connections.push(randomNode)
        }
      }
    })

    // Animation loop
    const animate = () => {
      if (!canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw and update nodes
      nodes.forEach((node) => {
        node.update()
        node.draw()
      })

      requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", setCanvasDimensions)
    }
  }, [])

  return (
    <motion.div
      className="w-full h-[400px] rounded-xl border border-border/50 bg-black/20 backdrop-blur-sm overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </motion.div>
  )
}

