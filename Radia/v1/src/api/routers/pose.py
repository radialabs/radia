"""
Pose estimation API endpoints
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_pose_service,
    get_hardware_service,
    get_current_user,
    require_auth
)
from src.services.pose_service import PoseService
from src.services.hardware_service import HardwareService
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response models
class PoseEstimationRequest(BaseModel):
    """Request model for pose estimation."""
    
    zone_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific zones to analyze (all zones if not specified)"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
    max_persons: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of persons to detect"
    )
    include_keypoints: bool = Field(
        default=True,
        description="Include detailed keypoint data"
    )
    include_segmentation: bool = Field(
        default=False,
        description="Include DensePose segmentation masks"
    )


class PersonPose(BaseModel):
    """Person pose data model."""
    
    person_id: str = Field(..., description="Unique person identifier")
    confidence: float = Field(..., description="Detection confidence score")
    bounding_box: Dict[str, float] = Field(..., description="Person bounding box")
    keypoints: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Body keypoints with coordinates and confidence"
    )
    segmentation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="DensePose segmentation data"
    )
    zone_id: Optional[str] = Field(
        default=None,
        description="Zone where person is detected"
    )
    activity: Optional[str] = Field(
        default=None,
        description="Detected activity"
    )
    timestamp: datetime = Field(..., description="Detection timestamp")


class PoseEstimationResponse(BaseModel):
    """Response model for pose estimation."""
    
    timestamp: datetime = Field(..., description="Analysis timestamp")
    frame_id: str = Field(..., description="Unique frame identifier")
    persons: List[PersonPose] = Field(..., description="Detected persons")
    zone_summary: Dict[str, int] = Field(..., description="Person count per zone")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HistoricalDataRequest(BaseModel):
    """Request model for historical pose data."""
    
    start_time: datetime = Field(..., description="Start time for data query")
    end_time: datetime = Field(..., description="End time for data query")
    zone_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter by specific zones"
    )
    aggregation_interval: Optional[int] = Field(
        default=300,
        ge=60,
        le=3600,
        description="Aggregation interval in seconds"
    )
    include_raw_data: bool = Field(
        default=False,
        description="Include raw detection data"
    )


# Endpoints
@router.get("/current", response_model=PoseEstimationResponse)
async def get_current_pose_estimation(
    request: PoseEstimationRequest = Depends(),
    pose_service: PoseService = Depends(get_pose_service),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get current pose estimation from WiFi signals."""
    try:
        logger.info(f"Processing pose estimation request from user: {current_user.get('id') if current_user else 'anonymous'}")
        
        # Get current pose estimation
        result = await pose_service.estimate_poses(
            zone_ids=request.zone_ids,
            confidence_threshold=request.confidence_threshold,
            max_persons=request.max_persons,
            include_keypoints=request.include_keypoints,
            include_segmentation=request.include_segmentation
        )
        
        return PoseEstimationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in pose estimation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pose estimation failed: {str(e)}"
        )


@router.post("/analyze", response_model=PoseEstimationResponse)
async def analyze_pose_data(
    request: PoseEstimationRequest,
    pose_service: PoseService = Depends(get_pose_service),
    current_user: Dict = Depends(require_auth)
):
    """Trigger pose analysis with custom parameters."""
    try:
        logger.info(f"Custom pose analysis requested by user: {current_user['id']}")
        
        result = await pose_service.estimate_poses(
            zone_ids=request.zone_ids,
            confidence_threshold=request.confidence_threshold,
            max_persons=request.max_persons,
            include_keypoints=request.include_keypoints,
            include_segmentation=request.include_segmentation
        )
        
        return PoseEstimationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in pose analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pose analysis failed: {str(e)}"
        )


@router.get("/zones/{zone_id}/occupancy")
async def get_zone_occupancy(zone_id: str):
    """Get current occupancy for a specific zone."""
    return {
        "zone_id": zone_id,
        "current_occupancy": 0,
        "max_occupancy": 10,
        "persons": [],
        "timestamp": datetime.utcnow(),
        "note": "No real-time CSI data available. Connect hardware to get live occupancy.",
    }


@router.get("/zones/summary")
async def get_zones_summary():
    """Get occupancy summary for all zones."""
    return {
        "timestamp": datetime.utcnow(),
        "total_persons": 0,
        "zones": {},
        "active_zones": 0,
        "note": "No real-time CSI data available. Connect hardware to get live occupancy.",
    }


@router.post("/historical")
async def get_historical_data(
    request: HistoricalDataRequest,
    current_user: Dict = Depends(require_auth)
):
    """Get historical pose estimation data."""
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="End time must be after start time")
    max_range = timedelta(days=7)
    if request.end_time - request.start_time > max_range:
        raise HTTPException(status_code=400, detail="Query range cannot exceed 7 days")
    return {
        "query": {
            "start_time": request.start_time,
            "end_time": request.end_time,
            "zone_ids": request.zone_ids,
            "aggregation_interval": request.aggregation_interval,
        },
        "data": [],
        "raw_data": None,
        "total_records": 0,
        "note": "No historical data available. A persistence backend must be configured.",
    }


@router.get("/activities")
async def get_detected_activities(
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of activities"),
):
    """Get recently detected activities."""
    return {
        "activities": [],
        "total_count": 0,
        "zone_id": zone_id,
        "note": "No activity data available without an active CSI stream.",
    }


@router.post("/calibrate")
async def calibrate_pose_system(
    background_tasks: BackgroundTasks,
    pose_service: PoseService = Depends(get_pose_service),
    hardware_service: HardwareService = Depends(get_hardware_service),
    current_user: Dict = Depends(require_auth)
):
    """Calibrate the pose estimation system."""
    try:
        logger.info(f"Pose system calibration initiated by user: {current_user['id']}")
        
        # Check if calibration is already in progress
        if await pose_service.is_calibrating():
            raise HTTPException(
                status_code=409,
                detail="Calibration already in progress"
            )
        
        # Start calibration process
        calibration_id = await pose_service.start_calibration()
        
        # Schedule background calibration task
        background_tasks.add_task(
            pose_service.run_calibration,
            calibration_id
        )
        
        return {
            "calibration_id": calibration_id,
            "status": "started",
            "estimated_duration_minutes": 5,
            "message": "Calibration process started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start calibration: {str(e)}"
        )


@router.get("/calibration/status")
async def get_calibration_status(
    pose_service: PoseService = Depends(get_pose_service),
    current_user: Dict = Depends(require_auth)
):
    """Get current calibration status."""
    try:
        status = await pose_service.get_calibration_status()
        
        return {
            "is_calibrating": status["is_calibrating"],
            "calibration_id": status.get("calibration_id"),
            "progress_percent": status.get("progress_percent", 0),
            "current_step": status.get("current_step"),
            "estimated_remaining_minutes": status.get("estimated_remaining_minutes"),
            "last_calibration": status.get("last_calibration")
        }
        
    except Exception as e:
        logger.error(f"Error getting calibration status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get calibration status: {str(e)}"
        )


@router.get("/stats")
async def get_pose_statistics(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    pose_service: PoseService = Depends(get_pose_service),
):
    """Get pose estimation statistics from current session."""
    status = await pose_service.get_status()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    return {
        "period": {"start_time": start_time, "end_time": end_time, "hours": hours},
        "statistics": status["statistics"],
    }