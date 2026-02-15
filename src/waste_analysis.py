def calculate_waste_risk(predicted, actual):
    """
    Calculates risk based on Percentage Error.
    Formula: (Predicted - Actual) / Actual * 100
    """
    if actual is None or actual == 0:
        return "Unknown", 0.0, 0.0

    difference = predicted - actual
    waste_percent = (difference / actual) * 100
    
    # risk Logic
    # positive % means overproduction (waste)
    # negative % means underproduction (lost sales)
    
    abs_waste = abs(waste_percent)
    
    if abs_waste < 10:
        risk_level = "Low Risk"
    elif 10 <= abs_waste < 25:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
        
    return risk_level, round(waste_percent, 1), difference