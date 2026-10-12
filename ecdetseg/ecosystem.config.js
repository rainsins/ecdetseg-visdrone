// ============================================================
// pm2 ecosystem for EdgeCrafter / ecdetseg experiments.
//
//   cd /home/rainsin/Projects/Career/Paper/EdgeCrafter/ecdetseg
//   VISDRONE_ROOT=/home/rainsin/Projects/Career/Paper/dataset/VisDrone \
//     pm2 start ecosystem.config.js --only p74u-all      # apps: p30u-all | p74u-all | p74u960-all
//   pm2 logs p74u-all
//   pm2 stop p74u-all        # -> SIGTERM -> run_all -> run_exp -> training
//
// ONE Arc GPU: each app runs its (init x seed) jobs SEQUENTIALLY.
// Output roots default to ecdetseg/outputs/{p30u,p74u,p74u_960}.
// autorestart=false because the guardian already does resume + retry + DONE.
// ============================================================
const path = require('path');

module.exports = {
  apps: [
    {
      name: 'p30u-all',
      script: 'scripts/run_all_p30u.sh',
      interpreter: 'bash',
      cwd: __dirname,
      autorestart: false,
      kill_timeout: 60000,          // let the stop-signal cascade to the training process
      stop_signal: 'SIGTERM',
      time: true,
      merge_logs: true,
      env: {
        // taken from the shell that runs `pm2 start` (no hardcoded /home/rainsin)
        VISDRONE_ROOT: process.env.VISDRONE_ROOT || '',
        WEIGHTS_DIR: process.env.WEIGHTS_DIR || '',
        P30U_RES: process.env.P30U_RES || '640',
        P30U_OUT_ROOT: process.env.P30U_OUT_ROOT || path.join(__dirname, 'outputs', 'p30u'),
        P30U_CONFIG: process.env.P30U_CONFIG || '',
        SEEDS: process.env.SEEDS || '42 43 44',
        INITS: process.env.INITS || 'e1 e2 e3',
        CONTINUE_ON_FAIL: process.env.CONTINUE_ON_FAIL || '0',
      },
    },
    {
      name: 'p74u-all',
      script: 'scripts/run_all_p30u.sh',
      interpreter: 'bash',
      cwd: __dirname,
      autorestart: false,
      kill_timeout: 60000,
      stop_signal: 'SIGTERM',
      time: true,
      merge_logs: true,
      env: {
        VISDRONE_ROOT: process.env.VISDRONE_ROOT || '',
        WEIGHTS_DIR: process.env.WEIGHTS_DIR || '',
        P74U_CONFIG: process.env.P74U_CONFIG || 'configs/ecdet/ecdet_s_p74u.yml',
        P74U_OUT_ROOT: process.env.P74U_OUT_ROOT || path.join(__dirname, 'outputs', 'p74u'),
        P30U_RES: process.env.P30U_RES || '640',
        SEEDS: process.env.SEEDS || '42 43 44',
        INITS: process.env.INITS || 'e1 e2 e3',
        CONTINUE_ON_FAIL: process.env.CONTINUE_ON_FAIL || '0',
      },
    },
    {
      name: 'p74u960-all',
      script: 'scripts/run_all_p30u.sh',
      interpreter: 'bash',
      cwd: __dirname,
      autorestart: false,
      kill_timeout: 60000,
      stop_signal: 'SIGTERM',
      time: true,
      merge_logs: true,
      env: {
        VISDRONE_ROOT: process.env.VISDRONE_ROOT || '',
        WEIGHTS_DIR: process.env.WEIGHTS_DIR || '',
        P74U_CONFIG: process.env.P74U_CONFIG || 'configs/ecdet/ecdet_s_p74u_960.yml',
        P74U_OUT_ROOT: process.env.P74U_OUT_ROOT || path.join(__dirname, 'outputs', 'p74u_960'),
        SEEDS: process.env.SEEDS || '42 43 44',
        INITS: process.env.INITS || 'e1 e2 e3',
        CONTINUE_ON_FAIL: process.env.CONTINUE_ON_FAIL || '0',
      },
    },

    // Single experiment (uncomment / use the CLI one-liner in the reply):
    // {
    //   name: 'p30u-e1-s42',
    //   script: 'scripts/run_exp.sh',
    //   interpreter: 'bash',
    //   args: 'e1 42',
    //   cwd: __dirname,
    //   autorestart: false,
    //   kill_timeout: 60000,
    //   stop_signal: 'SIGTERM',
    //   env: { VISDRONE_ROOT: process.env.VISDRONE_ROOT || '' },
    // },
  ],
};
