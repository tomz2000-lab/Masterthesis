"""
Pydantic schemas for validating and constraining LLM action outputs
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, Literal, Union
import json

class BaseAction(BaseModel):
    """Base action schema"""
    type: str

class OfferAction(BaseModel):
    """Price bargaining offer action"""
    type: Literal["offer"] = Field(..., description="Must be exactly 'offer'")
    price: float = Field(..., ge=0, description="Price in euros, must be positive")

class AcceptAction(BaseModel):
    """Accept action for any game type"""
    type: Literal["accept"] = Field(..., description="Must be exactly 'accept'")

class CounterAction(BaseModel):
    """Price bargaining counter-offer action"""
    type: Literal["counter"] = Field(..., description="Must be exactly 'counter'")
    price: float = Field(..., ge=0, description="Price in euros, must be positive")

class RejectAction(BaseModel):
    """Reject action for any game type"""
    type: Literal["reject"] = Field(..., description="Must be exactly 'reject'")

class ResourceProposalAction(BaseModel):
    """Resource allocation proposal with gpu_hours and bandwidth"""
    type: Literal["propose"] = Field(..., description="Must be exactly 'propose'")
    gpu_hours: float = Field(..., ge=0, description="GPU hours requested")
    bandwidth: float = Field(..., ge=0, description="Bandwidth requested")

class ProposeTradeAction(BaseModel):
    """Alternative resource allocation trade proposal"""
    type: Literal["propose_trade"] = Field(..., description="Must be exactly 'propose_trade'")
    offer: Dict[str, float] = Field(..., description="Resources offered")
    request: Dict[str, float] = Field(..., description="Resources requested")

class IntegrativeProposalAction(BaseModel):
    """Integrative negotiation proposal with nested structure"""
    type: Literal["propose"] = Field(..., description="Must be exactly 'propose'")
    proposal: Dict[str, Any] = Field(..., description="Nested proposal with server_room, meeting_access, cleaning, branding")

# Union of all possible actions
GameAction = Union[
    OfferAction,
    AcceptAction, 
    CounterAction,
    RejectAction,
    ResourceProposalAction,
    ProposeTradeAction,
    IntegrativeProposalAction
]

def validate_and_constrain_action(raw_response: str, game_type: str) -> Dict[str, Any]:
    """
    Validate and constrain LLM response to proper action format
    
    Args:
        raw_response: Raw JSON string from LLM
        game_type: Type of game (price_bargaining, resource_allocation, etc.)
    
    Returns:
        Validated and constrained action dictionary
    
    Raises:
        ValueError: If response cannot be validated/constrained
    """
    try:
        # Parse JSON
        parsed = json.loads(raw_response.strip())
        
        # Validate based on game type and action type
        action_type = parsed.get("type", "")
        
        if action_type == "accept":
            action = AcceptAction(**parsed)
        elif action_type == "reject":
            action = RejectAction(**parsed)
        elif action_type == "offer" and game_type in ["price_bargaining", "company_car"]:
            action = OfferAction(**parsed)
        elif action_type == "counter" and game_type in ["price_bargaining", "company_car"]:
            action = CounterAction(**parsed)
        elif action_type == "propose" and game_type == "resource_allocation":
            action = ResourceProposalAction(**parsed)
        elif action_type == "propose_trade" and game_type == "resource_allocation":
            action = ProposeTradeAction(**parsed)
        elif action_type == "propose" and game_type == "integrative_negotiations":
            action = IntegrativeProposalAction(**parsed)
        else:
            # Try to auto-correct common mistakes
            corrected = auto_correct_action(parsed, game_type)
            if corrected:
                return corrected
            raise ValueError(f"Invalid action type '{action_type}' for game '{game_type}'")
        
        return action.dict()
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Action validation failed: {e}")

def auto_correct_action(parsed: Dict[str, Any], game_type: str) -> Optional[Dict[str, Any]]:
    """
    Auto-correct common LLM mistakes in action format
    
    Returns:
        Corrected action dict or None if cannot be corrected
    """
    action_type = parsed.get("type", "").lower()
    
    # Auto-correct common acceptance variations
    if any(word in action_type for word in ["accept", "agree", "yes"]):
        return {"type": "accept"}
    
    # Auto-correct common rejection variations
    if any(word in action_type for word in ["reject", "decline", "no"]):
        return {"type": "reject"}
    
    # Auto-correct offer variations for price bargaining
    if game_type in ["price_bargaining", "company_car"] and any(word in action_type for word in ["offer", "bid", "propose"]):
        price = parsed.get("price") or parsed.get("amount") or parsed.get("value")
        if price is not None:
            return {"type": "offer", "price": float(price)}
    
    # Auto-correct counter variations for price bargaining
    if game_type in ["price_bargaining", "company_car"] and any(word in action_type for word in ["counter", "counteroffer"]):
        price = parsed.get("price") or parsed.get("amount") or parsed.get("value")
        if price is not None:
            return {"type": "counter", "price": float(price)}
    
    # Auto-correct resource allocation variations
    if game_type == "resource_allocation" and any(word in action_type for word in ["trade", "propose", "offer"]):
        # Try standard format first (gpu_hours, bandwidth)
        gpu_hours = parsed.get("gpu_hours") or parsed.get("gpu") or parsed.get("x")
        bandwidth = parsed.get("bandwidth") or parsed.get("y") or parsed.get("bw")
        
        if gpu_hours is not None and bandwidth is not None:
            return {"type": "propose", "gpu_hours": float(gpu_hours), "bandwidth": float(bandwidth)}
        
        # Try trade format as fallback
        offer = parsed.get("offer") or parsed.get("give") or {}
        request = parsed.get("request") or parsed.get("want") or parsed.get("ask") or {}
        if offer and request:
            return {"type": "propose_trade", "offer": offer, "request": request}
    
    # Auto-correct integrative negotiations variations
    if game_type == "integrative_negotiations" and any(word in action_type for word in ["propose", "offer"]):
        # Check if it's already in the correct nested format
        proposal = parsed.get("proposal")
        if proposal and isinstance(proposal, dict):
            # Auto-correct common invalid values to nearest valid options
            corrected_proposal = {}
            
            # Auto-correct server_room to nearest valid value
            if "server_room" in proposal:
                server_room = proposal["server_room"]
                if server_room <= 75:
                    corrected_proposal["server_room"] = 50
                elif server_room <= 125:
                    corrected_proposal["server_room"] = 100
                else:
                    corrected_proposal["server_room"] = 150
            
            # Auto-correct meeting_access to nearest valid value  
            if "meeting_access" in proposal:
                meeting_access = proposal["meeting_access"]
                if meeting_access <= 3:
                    corrected_proposal["meeting_access"] = 2
                elif meeting_access <= 5:
                    corrected_proposal["meeting_access"] = 4
                else:
                    corrected_proposal["meeting_access"] = 7
            
            # Auto-correct cleaning values
            if "cleaning" in proposal:
                cleaning = str(proposal["cleaning"]).lower()
                if "it" in cleaning:
                    corrected_proposal["cleaning"] = "IT"
                elif "shared" in cleaning:
                    corrected_proposal["cleaning"] = "Shared"
                else:
                    corrected_proposal["cleaning"] = "Outsourced"
            
            # Auto-correct branding values
            if "branding" in proposal:
                branding = str(proposal["branding"]).lower()
                if "minimal" in branding:
                    corrected_proposal["branding"] = "Minimal"
                elif "moderate" in branding:
                    corrected_proposal["branding"] = "Moderate"
                else:
                    corrected_proposal["branding"] = "Prominent"
            
            # Use original values for any keys not corrected
            for key, value in proposal.items():
                if key not in corrected_proposal:
                    corrected_proposal[key] = value
                    
            return {"type": "propose", "proposal": corrected_proposal}
        
        # Try to extract proposal from flat format
        server_room = parsed.get("server_room") 
        meeting_access = parsed.get("meeting_access")
        cleaning = parsed.get("cleaning")
        branding = parsed.get("branding")
        
        if all(x is not None for x in [server_room, meeting_access, cleaning, branding]):
            # Apply same corrections to flat format
            corrected_server_room = 50 if server_room <= 75 else (100 if server_room <= 125 else 150)
            corrected_meeting_access = 2 if meeting_access <= 3 else (4 if meeting_access <= 5 else 7)
            
            return {"type": "propose", "proposal": {
                "server_room": corrected_server_room,
                "meeting_access": corrected_meeting_access, 
                "cleaning": str(cleaning).title(),
                "branding": str(branding).title()
            }}
    
    return None
