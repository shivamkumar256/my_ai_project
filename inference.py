def predict(request):
    """
    Required function for OpenEnv validation
    """
    try:
        data = request.get("input", {})
        
        return {
            "status": "success",
            "output": data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
