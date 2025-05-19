from chat_graph import create_chat_graph
import graphviz
import os

def visualize_chat_graph():
    """Create a visual representation of the chat processing graph."""
    # Create the graph
    graph = create_chat_graph()
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Chat Processing Graph')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Add nodes with custom styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Add the nodes
    dot.node('create_chunks', 'Create Chunks\n(Split chat into manageable pieces)')
    dot.node('process_chunk', 'Process Chunk\n(Analyze with LLM)')
    dot.node('combine_responses', 'Combine Responses\n(Merge all analyses)')
    
    # Add edges
    dot.edge('create_chunks', 'process_chunk', label='Start Processing')
    dot.edge('process_chunk', 'process_chunk', label='More Chunks?')
    dot.edge('process_chunk', 'combine_responses', label='All Chunks Processed')
    
    # Save the graph
    output_path = 'chat_graph'
    dot.render(output_path, format='png', cleanup=True)
    print(f"Graph visualization saved as {output_path}.png")
    
    # Also save as PDF for better quality
    dot.render(output_path, format='pdf', cleanup=True)
    print(f"Graph visualization saved as {output_path}.pdf")

if __name__ == "__main__":
    visualize_chat_graph() 