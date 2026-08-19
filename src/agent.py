import json
from openai import OpenAI
from physics import evaluate_macroscopic_fluid_strain

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "evaluate_macroscopic_fluid_strain",
        "description": "Calculates physical crowd fluid strain and returns current risk state.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "string", "description": "Selected camera identifier"}
            },
            "required": ["camera_id"]
        }
    }
}]

def run_triage_agent(db_session, camera_id: str) -> str:
    messages = [
        {"role": "system", "content": "You are the Ghent Crowd Safety Dispatcher. Always call 'evaluate_macroscopic_fluid_strain' to get ground truth metrics before drafting your report."},
        {"role": "user", "content": f"Assess crowd safety for camera location: {camera_id}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # Handle LLM Tool Call
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "evaluate_macroscopic_fluid_strain":
        args = json.loads(tool_call.function.arguments)
        
        # Execute local Python physics function
        physics_output = evaluate_macroscopic_fluid_strain(db_session, args["camera_id"])
        
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": physics_output
        })
        
        # Get final response from LLM
        final_response = client.chat.completions.create(model="gpt-4o", messages=messages)
        return final_response.choices[0].message.content