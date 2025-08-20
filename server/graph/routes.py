from fastapi import APIRouter, Depends, Query, HTTPException, Body
from auth.routes import get_current_user
from auth.models import UserProfile
from typing import Optional, List, Dict, Any
import networkx as nx
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Import the knowledge graph from the advanced RAG engine
from chat.advanced_rag import rag_engine

@router.get("/stats")
async def get_graph_stats(user: UserProfile = Depends(get_current_user)):
    """Get knowledge graph statistics"""
    
    if "graph_query" not in user.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to access knowledge graph"
        )
    
    try:
        graph = rag_engine.knowledge_graph
        
        # Calculate basic statistics
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        
        # Calculate more advanced metrics
        density = nx.density(graph) if num_nodes > 0 else 0
        
        # Get degree distribution
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        
        # Find most connected entities
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "user": user.username,
            "role": user.role,
            "timestamp": datetime.utcnow().isoformat(),
            "graph_stats": {
                "total_nodes": num_nodes,
                "total_edges": num_edges,
                "density": round(density, 4),
                "average_degree": round(avg_degree, 2),
                "most_connected_entities": [
                    {"entity": entity, "connections": degree}
                    for entity, degree in top_entities
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve graph statistics"
        )

@router.get("/entities/{entity}")
async def get_entity_details(
    entity: str,
    user: UserProfile = Depends(get_current_user),
    depth: int = Query(default=1, ge=1, le=3, description="Relationship depth (1-3)")
):
    """Get detailed information about a specific entity and its relationships"""
    
    if "graph_query" not in user.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to query knowledge graph"
        )
    
    try:
        graph = rag_engine.knowledge_graph
        
        if entity not in graph:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity}' not found in knowledge graph"
            )
        
        # Get entity information
        entity_data = graph.nodes[entity]
        
        # Get neighbors up to specified depth
        subgraph = nx.ego_graph(graph, entity, radius=depth)
        
        # Extract relationships
        relationships = []
        for source, target, edge_data in subgraph.edges(data=True):
            relationships.append({
                "from": source,
                "to": target,
                "relationship_type": edge_data.get("type", "related"),
                "weight": edge_data.get("weight", 0.5),
                "source_document": edge_data.get("source", "unknown"),
                "last_updated": edge_data.get("last_updated", "unknown")
            })
        
        # Get neighboring entities
        direct_neighbors = list(graph.neighbors(entity))
        
        return {
            "entity": entity,
            "entity_data": entity_data,
            "direct_neighbors": direct_neighbors,
            "relationships": relationships,
            "subgraph_info": {
                "depth_explored": depth,
                "nodes_in_subgraph": len(subgraph.nodes()),
                "edges_in_subgraph": len(subgraph.edges())
            },
            "user": user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity details: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve entity information"
        )

@router.post("/query")
async def execute_graph_query(
    user: UserProfile = Depends(get_current_user),
    query_data: Dict[str, Any] = Body(...)
):
    """Execute complex graph queries"""
    
    if "graph_query" not in user.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to execute graph queries"
        )
    
    try:
        graph = rag_engine.knowledge_graph
        query_type = query_data.get("type")
        entities = query_data.get("entities", [])
        
        if not query_type:
            raise HTTPException(
                status_code=400,
                detail="Query type is required"
            )
        
        result = {}
        
        if query_type == "shortest_path":
            if len(entities) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Shortest path query requires at least 2 entities"
                )
            
            try:
                path = nx.shortest_path(graph, entities[0], entities[1])
                path_length = len(path) - 1
                
                # Get edge details along the path
                path_details = []
                for i in range(len(path) - 1):
                    source, target = path[i], path[i + 1]
                    edge_data = graph[source][target] if graph.has_edge(source, target) else {}
                    path_details.append({
                        "from": source,
                        "to": target,
                        "relationship": edge_data.get("type", "related"),
                        "weight": edge_data.get("weight", 0.5)
                    })
                
                result = {
                    "path": path,
                    "path_length": path_length,
                    "path_details": path_details
                }
                
            except nx.NetworkXNoPath:
                result = {
                    "path": None,
                    "message": f"No path found between '{entities[0]}' and '{entities[1]}'"
                }
        
        elif query_type == "common_neighbors":
            if len(entities) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Common neighbors query requires at least 2 entities"
                )
            
            common = list(nx.common_neighbors(graph, entities[0], entities[1]))
            result = {
                "entities": entities[:2],
                "common_neighbors": common,
                "count": len(common)
            }
        
        elif query_type == "node_centrality":
            entity = entities[0] if entities else None
            if not entity:
                raise HTTPException(
                    status_code=400,
                    detail="Node centrality query requires an entity"
                )
            
            if entity not in graph:
                raise HTTPException(
                    status_code=404,
                    detail=f"Entity '{entity}' not found"
                )
            
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(graph).get(entity, 0)
            
            try:
                betweenness_centrality = nx.betweenness_centrality(graph).get(entity, 0)
                closeness_centrality = nx.closeness_centrality(graph).get(entity, 0)
            except:
                betweenness_centrality = 0
                closeness_centrality = 0
            
            result = {
                "entity": entity,
                "centrality_measures": {
                    "degree": round(degree_centrality, 4),
                    "betweenness": round(betweenness_centrality, 4),
                    "closeness": round(closeness_centrality, 4)
                }
            }
        
        elif query_type == "subgraph":
            if not entities:
                raise HTTPException(
                    status_code=400,
                    detail="Subgraph query requires at least one entity"
                )
            
            # Create subgraph containing all specified entities
            subgraph_nodes = set(entities)
            
            # Add direct neighbors
            for entity in entities:
                if entity in graph:
                    subgraph_nodes.update(graph.neighbors(entity))
            
            subgraph = graph.subgraph(subgraph_nodes)
            
            result = {
                "requested_entities": entities,
                "subgraph_nodes": list(subgraph_nodes),
                "subgraph_stats": {
                    "nodes": len(subgraph.nodes()),
                    "edges": len(subgraph.edges()),
                    "density": nx.density(subgraph)
                }
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported query type: {query_type}"
            )
        
        return {
            "query_type": query_type,
            "executed_by": user.username,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph query execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Graph query execution failed: {str(e)}"
        )

@router.get("/search")
async def search_entities(
    user: UserProfile = Depends(get_current_user),
    query: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(default=10, ge=1, le=50)
):
    """Search for entities in the knowledge graph"""
    
    if "graph_query" not in user.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to search knowledge graph"
        )
    
    try:
        graph = rag_engine.knowledge_graph
        query_lower = query.lower()
        
        # Search entities by name
        matching_entities = []
        for entity in graph.nodes():
            if query_lower in entity.lower():
                entity_data = graph.nodes[entity]
                degree = graph.degree(entity)
                
                matching_entities.append({
                    "entity": entity,
                    "degree": degree,
                    "type": entity_data.get("type", "unknown"),
                    "frequency": entity_data.get("frequency", 1)
                })
        
        # Sort by relevance (degree and frequency)
        matching_entities.sort(
            key=lambda x: (x["degree"], x["frequency"]),
            reverse=True
        )
        
        return {
            "query": query,
            "results_count": len(matching_entities),
            "results": matching_entities[:limit],
            "user": user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Entity search failed"
        )

@router.get("/export")
async def export_graph_data(
    user: UserProfile = Depends(get_current_user),
    format: str = Query(default="json", regex="^(json|gexf|graphml)$")
):
    """Export knowledge graph data"""
    
    if user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only administrators can export graph data"
        )
    
    try:
        graph = rag_engine.knowledge_graph
        
        if format == "json":
            # Convert to JSON format
            data = {
                "nodes": [
                    {"id": node, "attributes": graph.nodes[node]}
                    for node in graph.nodes()
                ],
                "edges": [
                    {
                        "source": source,
                        "target": target,
                        "attributes": graph[source][target]
                    }
                    for source, target in graph.edges()
                ],
                "metadata": {
                    "exported_by": user.username,
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "total_nodes": len(graph.nodes()),
                    "total_edges": len(graph.edges())
                }
            }
            
            return data
        
        else:
            # For other formats, you'd implement the specific export logic
            raise HTTPException(
                status_code=501,
                detail=f"Export format '{format}' not yet implemented"
            )
    
    except Exception as e:
        logger.error(f"Graph export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Graph export failed"
        )

@router.get("/health")
async def graph_health():
    """Health check for knowledge graph services"""
    
    try:
        graph = rag_engine.knowledge_graph
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "knowledge_graph": "operational",
                "networkx": "operational",
                "entity_processing": "operational"
            },
            "graph_info": {
                "total_nodes": len(graph.nodes()),
                "total_edges": len(graph.edges()),
                "is_directed": graph.is_directed()
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
