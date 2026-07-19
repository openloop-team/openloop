CREATE INDEX broker_generation_checkpoint_recovery
ON broker_generations (job_id, generation)
WHERE state IN ('quiescing', 'quiesced', 'releasing');

CREATE INDEX broker_generation_expired_running_recovery
ON broker_generations (execution_lease_deadline, job_id)
WHERE state = 'running';

CREATE INDEX broker_job_finalizing_recovery
ON broker_jobs (job_id)
WHERE state = 'finalizing';
