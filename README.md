# Testing the job

`env $(cat src/mms_blog_post.env skypilot-conf/debug.env | grep -v '^#' | xargs) bash -c 'uv run sky jobs launch src/mms_blog_post.job.yaml'`
