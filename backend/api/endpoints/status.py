"""Job status endpoint."""

from fastapi import APIRouter, HTTPException, Query, Response, status

from backend.api.schemas import CancelJobResponse, JobStatus
from backend.session.manager import session_manager

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    from backend.runtime.job_event_monitor import sync_job_events_once

    await sync_job_events_once(job_id)
    job = session_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    # A saved/previous export does not mean the current generation finished.
    normalized_status = job.status

    return JobStatus(
        status=normalized_status,
        progress=job.progress,
        message=job.message,
        slides_completed=job.slides_completed,
        total_slides=job.total_slides,
        output_path=job.output_path,
        error=None if normalized_status == "complete" else job.error,
        provider=job.provider,
        model=job.model_name,
        base_url=job.base_url,
    )


@router.get("/status/{job_id}/events")
async def get_job_events(job_id: str, since_seq: int = Query(default=0, ge=0)) -> dict:
    """Return retained job events for deterministic result-page hydration."""
    from backend.runtime.job_event_monitor import sync_job_events_once

    await sync_job_events_once(job_id)
    job = session_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
    return {
        "job_id": job_id,
        "last_seq": job.last_seq,
        "events": session_manager.get_events_after(job_id, since_seq),
    }


@router.post("/status/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(job_id: str) -> CancelJobResponse:
    job = session_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job.status in {"complete", "error", "cancelled"}:
        return CancelJobResponse(job_id=job_id, status=job.status)

    cancelled = session_manager.cancel_job(job_id)
    if not cancelled:
        session_manager.mark_job_cancelled(job_id)
        await session_manager.flush_now()
        return CancelJobResponse(job_id=job_id, status="cancelled")
    job = session_manager.get_job(job_id)
    if job is not None and job.status == "cancelled":
        await session_manager.flush_now()
        return CancelJobResponse(job_id=job_id, status="cancelled")

    return CancelJobResponse(job_id=job_id, status="cancelling")


@router.delete("/status/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str) -> Response:
    deleted = session_manager.delete_job(job_id, delete_files=True)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
    await session_manager.flush_now()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
