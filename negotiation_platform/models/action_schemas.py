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
    """Integrative negotiation proposal"""
    type: Literal["propose"] = Field(..., description="Must be exactly 'propose'")
    server_room: int = Field(..., ge=50, le=150, description="Server room size in sqm")
    meeting_access: int = Field(..., ge=2, le=7, description="Meeting access days per week")
    cleaning: Literal["shared", "it", "outsourced"] = Field(..., description="Cleaning responsibility")
    branding: Literal["minimal", "moderate", "prominent"] = Field(..., description="Branding visibility")

# Union of all possible actions
GameAction = Union[
    OfferAction,
    AcceptAction, 
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
        elif action_type == "offer" and game_type in ["price_bargaining", "company_car"]:
            action = OfferAction(**parsed)
        elif action_type == "propose" and game_type == "resource_allocation":
            action = ResourceProposalAction(**parsed)
        elif action_type == "propose_trade" and game_type == "resource_allocation":
            action = ProposeTradeAction(**parsed)
        elif action_type == "propose" and game_type == "integrative":
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
    
    # Auto-correct offer variations for price bargaining
    if game_type in ["price_bargaining", "company_car"] and any(word in action_type for word in ["offer", "bid", "propose"]):
        price = parsed.get("price") or parsed.get("amount") or parsed.get("value")
        if price is not None:
            return {"type": "offer", "price": float(price)}
    
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
    
    return None
