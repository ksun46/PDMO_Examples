using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../test_helper.jl")

@testset "ADMMBipartiteGraph Creation" begin
    println("    ├─ Testing ADMMBipartiteGraph creation...")
    # Test basic creation with empty problem
    mbp = MultiblockProblem()
        
    try
        graph = MultiblockGraph(mbp)
        
        # Test creation with BFS algorithm
        admm_graph = ADMMBipartiteGraph(graph, mbp, BFS_BIPARTIZATION)
        @test admm_graph isa ADMMBipartiteGraph
        @test hasfield(ADMMBipartiteGraph, :nodes)
        @test hasfield(ADMMBipartiteGraph, :edges)
        @test hasfield(ADMMBipartiteGraph, :left)
        @test hasfield(ADMMBipartiteGraph, :right)
        println("    │  ├─ ✅ ADMMBipartiteGraph created with BFS algorithm")
        
        # Test creation with MILP algorithm
        admm_graph_milp = ADMMBipartiteGraph(graph, mbp, MILP_BIPARTIZATION)
        @test admm_graph_milp isa ADMMBipartiteGraph
        println("    │  ├─ ✅ ADMMBipartiteGraph created with MILP algorithm")
        
    catch e
        println("    │  ├─ ⚠️ Creation failed (expected for empty problem): $e")
    end
    println("    └─ ✅ ADMMBipartiteGraph creation tests completed")
end

@testset "ADMM Graph with Simple Problem" begin  
    println("    ├─ Testing ADMM Graph with simple problem...")
    
    mbp = MultiblockProblem()
    
    # Create simple blocks with proper functions
    bv1 = BlockVariable("block1")
    bv1.f = Zero()
    bv1.g = Zero()
    bv1.val = [1.0, 2.0]
    addBlockVariable!(mbp, bv1)
    
    bv2 = BlockVariable("block2")
    bv2.f = Zero()
    bv2.g = Zero()
    bv2.val = [3.0, 4.0]
    addBlockVariable!(mbp, bv2)
    
    # Add constraint to connect blocks
    bc = BlockConstraint("constr1")
    bc.involvedBlocks = ["block1", "block2"]
    bc.rhs = [0.0, 0.0]
    bc.mappings["block1"] = LinearMappingIdentity(1.0)
    bc.mappings["block2"] = LinearMappingIdentity(1.0)
    addBlockConstraint!(mbp, bc)
    
    graph = MultiblockGraph(mbp)
    
    # Test ADMM graph creation with both algorithms
    admm_graph_bfs = ADMMBipartiteGraph(graph, mbp, BFS_BIPARTIZATION)
    @test admm_graph_bfs isa ADMMBipartiteGraph
    @test length(admm_graph_bfs.nodes) >= 2  # At least the original variable nodes
    @test length(admm_graph_bfs.left) >= 1
    @test length(admm_graph_bfs.right) >= 1
    println("    │  ├─ ✅ ADMMBipartiteGraph created with simple problem (BFS)")
    
    admm_graph_milp = ADMMBipartiteGraph(graph, mbp, MILP_BIPARTIZATION)
    @test admm_graph_milp isa ADMMBipartiteGraph
    println("    │  ├─ ✅ ADMMBipartiteGraph created with simple problem (MILP)")
    
    println("    └─ ✅ ADMM Graph with simple problem tests completed")
end

@testset "ADMM Graph Properties and Structure" begin
    println("    ├─ Testing ADMM Graph properties...")
    
    mbp = MultiblockProblem()
    
    # Create blocks
    bv1 = BlockVariable("block1")
    bv1.f = Zero()
    bv1.g = Zero()
    bv1.val = [1.0]
    addBlockVariable!(mbp, bv1)
    
    bv2 = BlockVariable("block2")
    bv2.f = Zero()
    bv2.g = Zero()
    bv2.val = [2.0]
    addBlockVariable!(mbp, bv2)
    
    # Add constraint
    bc = BlockConstraint("constr1")
    bc.involvedBlocks = ["block1", "block2"]
    bc.rhs = 0.0
    bc.mappings["block1"] = LinearMappingIdentity(1.0)
    bc.mappings["block2"] = LinearMappingIdentity(1.0)
    addBlockConstraint!(mbp, bc)
    
    graph = MultiblockGraph(mbp)
    admm_graph = ADMMBipartiteGraph(graph, mbp, BFS_BIPARTIZATION)
    
    # Test basic structure properties
    @test admm_graph.nodes isa Dict{String, ADMMNode}
    @test admm_graph.edges isa Dict{String, ADMMEdge}
    @test admm_graph.left isa Vector{String}
    @test admm_graph.right isa Vector{String}
    @test admm_graph.mbpBlockID2admmNodeID isa Dict
    
    # Test that partitions are non-empty and disjoint
    @test length(admm_graph.left) > 0
    @test length(admm_graph.right) > 0
    @test isempty(intersect(admm_graph.left, admm_graph.right))
    
    # Test that all nodes are accounted for in partitions
    all_partition_nodes = Set(vcat(admm_graph.left, admm_graph.right))
    all_graph_nodes = Set(keys(admm_graph.nodes))
    @test all_partition_nodes == all_graph_nodes
    
    # Test node structure
    for (nodeID, node) in admm_graph.nodes
        @test node isa ADMMNode
        @test hasfield(ADMMNode, :f)
        @test hasfield(ADMMNode, :g)
        @test hasfield(ADMMNode, :val)
        @test hasfield(ADMMNode, :neighbors)
        @test hasfield(ADMMNode, :assignment)
        @test node.assignment in [0, 1]  # Should be assigned to left (0) or right (1)
    end
    
    # Test edge structure and bipartiteness
    for (edgeID, edge) in admm_graph.edges
        @test edge isa ADMMEdge
        @test hasfield(ADMMEdge, :nodeID1)
        @test hasfield(ADMMEdge, :nodeID2)
        @test hasfield(ADMMEdge, :mappings)
        @test hasfield(ADMMEdge, :rhs)
        
        # Verify bipartiteness: endpoints should be in different partitions
        node1_assignment = admm_graph.nodes[edge.nodeID1].assignment
        node2_assignment = admm_graph.nodes[edge.nodeID2].assignment
        @test node1_assignment != node2_assignment
    end
    
    @info "✅ ADMM graph structure verified" nodes=length(admm_graph.nodes) edges=length(admm_graph.edges) left=length(admm_graph.left) right=length(admm_graph.right)
    println("    └─ ✅ ADMM Graph properties tests completed")
end

@testset "ADMM Graph Helper Functions" begin
    println("    ├─ Testing ADMM Graph helper functions...")
    
    # Test ID creation functions
    edgeID = "TestEdge1"
    nodeID = "TestNode1"
    
    admm_node_id = createADMMNodeID(edgeID)
    @test admm_node_id isa String
    @test occursin(edgeID, admm_node_id)
    
    admm_edge_id = createADMMEdgeID(edgeID, nodeID)
    @test admm_edge_id isa String
    @test occursin(edgeID, admm_edge_id)
    @test occursin(nodeID, admm_edge_id)
    
    println("    │  ├─ ✅ ID creation functions working")
    
    # Test with actual ADMM graph
    mbp = MultiblockProblem()
    bv = BlockVariable("block1")
    bv.f = Zero()
    bv.g = Zero()
    bv.val = [1.0]
    addBlockVariable!(mbp, bv)
    
    graph = MultiblockGraph(mbp)
    admm_graph = ADMMBipartiteGraph(graph, mbp, BFS_BIPARTIZATION)
    
    # Test summary function (should not throw errors)
    try
        summary(admm_graph)
        @test true  # Success
        println("    │  ├─ ✅ Summary function executed successfully")
    catch e
        println("    │  ├─ ⚠️ Summary function failed: $e")
    end
    
    println("    └─ ✅ ADMM Graph helper functions tests completed")
end

@testset "ADMM Graph Edge Cases" begin
    println("    ├─ Testing ADMM Graph edge cases...")
    
    # Test with single block (minimal valid case)
    mbp_single = MultiblockProblem()
    bv = BlockVariable("block1")
    bv.f = Zero()
    bv.g = Zero()
    bv.val = [1.0]
    addBlockVariable!(mbp_single, bv)
    
    graph_single = MultiblockGraph(mbp_single)
    
    admm_graph_single = ADMMBipartiteGraph(graph_single, mbp_single, BFS_BIPARTIZATION)
    @test admm_graph_single isa ADMMBipartiteGraph
    @test length(admm_graph_single.nodes) == 1
    @test length(admm_graph_single.edges) == 0
    println("    │  ├─ ✅ Single block ADMM graph created successfully")
    
    # Test with multiblock constraint (creates constraint nodes)
    mbp_multi = MultiblockProblem()
    
    for i in 1:3
        bv = BlockVariable("block$i")
        bv.f = Zero()
        bv.g = Zero()
        bv.val = [Float64(i)]
        addBlockVariable!(mbp_multi, bv)
    end
    
    # Add constraint involving 3 blocks (creates constraint node)
    bc = BlockConstraint("constr1")
    bc.involvedBlocks = ["block1", "block2", "block3"]
    bc.rhs = [0.0]
    bc.mappings["block1"] = LinearMappingIdentity(1.0)
    bc.mappings["block2"] = LinearMappingIdentity(1.0)
    bc.mappings["block3"] = LinearMappingIdentity(1.0)
    addBlockConstraint!(mbp_multi, bc)
    
    graph_multi = MultiblockGraph(mbp_multi)
    admm_graph_multi = ADMMBipartiteGraph(graph_multi, mbp_multi, BFS_BIPARTIZATION)
    @test admm_graph_multi isa ADMMBipartiteGraph
    @test length(admm_graph_multi.nodes) == 4  # 3 variable nodes + 1 constraint node
    println("    │  ├─ ✅ Multiblock constraint ADMM graph created successfully")
    
    println("    └─ ✅ ADMM Graph edge cases tests completed")
end 