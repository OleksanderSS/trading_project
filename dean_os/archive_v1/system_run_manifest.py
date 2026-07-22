import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SystemRunManifest(BaseModel):
    """
    A manifest representing a single daily execution run of the pipeline.
    This guarantees that every run is reproducible and audited.
    """
    run_id: str = Field(description="Unique ID for this run, generated from timestamp")
    as_of: str = Field(description="The point-in-time this run represents")
    domain_id: str = Field(description="The domain being executed")
    
    status: str = Field(default="pending", description="pending, success, or failed")
    
    collector_status: Dict[str, Any] = Field(default_factory=dict, description="Health of upstream collectors")
    input_hashes: Dict[str, str] = Field(default_factory=dict, description="Hashes of pipeline contexts used")
    event_packet_id: Optional[str] = Field(default=None, description="The ID of the generated WorldModelEventLearningPacket")
    review_gate_id: Optional[str] = Field(default=None, description="The ID of the WorldModelReplayReviewGate")
    gate_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of the gate's decision")
    
    audit_log: List[str] = Field(default_factory=list, description="Chronological log of major steps")
    
    @classmethod
    def initialize(cls, as_of: str, domain_id: str) -> "SystemRunManifest":
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"sysrun_{now}_{domain_id}"
        manifest = cls(run_id=run_id, as_of=as_of, domain_id=domain_id)
        manifest.log(f"Run initialized at {now}")
        return manifest
        
    def log(self, message: str) -> None:
        """Appends a message to the daily audit log."""
        self.audit_log.append(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {message}")

    def register_pipeline_context(self, context_id: str, context_dict: dict) -> None:
        """Records the hash of an input context to ensure provenance."""
        content_hash = hashlib.sha256(json.dumps(context_dict, sort_keys=True).encode()).hexdigest()
        self.input_hashes[context_id] = content_hash
        self.log(f"Registered context {context_id} with hash {content_hash[:8]}...")
        
    def set_collector_status(self, is_healthy: bool, details: str = "") -> None:
        self.collector_status = {
            "healthy": is_healthy,
            "details": details
        }
        status_str = "HEALTHY" if is_healthy else "FAILED"
        self.log(f"Collectors verified. Status: {status_str}. {details}")
        if not is_healthy:
            self.status = "failed"
            
    def set_event_packet(self, packet_id: str) -> None:
        self.event_packet_id = packet_id
        self.log(f"Event packet generated: {packet_id}")
        
    def set_review_gate(self, gate_id: str, summary: dict) -> None:
        self.review_gate_id = gate_id
        self.gate_summary = summary
        self.log(f"Review gate completed: {gate_id} with status {summary.get('gate_status')}")
        
    def mark_completed(self) -> None:
        if self.status != "failed":
            self.status = "success"
            self.log("Run completed successfully.")
        else:
            self.log("Run completed with failures.")

    def as_json(self) -> str:
        return self.model_dump_json(indent=2)
