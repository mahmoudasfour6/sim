# Run once in Windows PowerShell (not WSL) to schedule the nightly regression at 02:30.
# The task runs inside the default WSL distribution; the laptop must be on (or set "Wake the computer to run
# this task" in Task Scheduler > task > Conditions). The newest log is copied to %USERPROFILE%\sim_backup.
$action  = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-e bash -lc '~/sim_workspace/tools/nightly.sh'"
$trigger = New-ScheduledTaskTrigger -Daily -At 2:30am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName "sim_workspace nightly" -Action $action -Trigger $trigger -Settings $settings -Force
Write-Host "Registered. Run now with:  Start-ScheduledTask -TaskName 'sim_workspace nightly'"
