from fastapi import FastAPI, APIRouter, HTTPException

from lib.generators import *
from lib.mock_data import *
from lib.type_defs import *

api_router = APIRouter(prefix="/api")


def validate_dates(from_date: dt.date, to_date: dt.date):
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")


# region get injury types, treatment types, etc

@api_router.get("/types/injuries", response_model=List[InjuryType], tags=["types"])
def get_injury_types():
    return injury_types


@api_router.get("/types/treatments", response_model=List[TreatmentType], tags=["types"])
def get_treatment_types():
    return treatment_types


# endregion get injury types, treatment types, etc

# region get nurses, residents, etc

@api_router.get("/people/nurses", response_model=List[NurseInfo], tags=["people"])
def get_nurses():
    return mock_nurses


@api_router.get("/people/residents", response_model=List[ResidentInfo], tags=["people"])
def get_residents():
    return mock_residents


# endregion get nurses, residents, etc

# region get facility, floor, etc

@api_router.get("/facilities/facilities", response_model=List[FacilityInfo], tags=["facility"])
def get_facilities():
    return mock_facilities


@api_router.get("/facilities/floors", response_model=List[FloorInfo], tags=["facility"])
def get_floors():
    return mock_floors


@api_router.get("/facilities/rooms", response_model=List[RoomInfo], tags=["facility"])
def get_rooms():
    return mock_rooms


@api_router.get("/facilities/beds", response_model=List[BedInfo], tags=["facility"])
def get_beds():
    return mock_beds


# endregion get facility, floor, etc

# region get events

@api_router.get("/events/resident-injury-events", response_model=List[ResidentInjuryEvent], tags=["events"])
def get_resident_injury_events(from_date: dt.date, to_date: dt.date):
    validate_dates(from_date, to_date)
    return get_events_date_range(from_date, to_date, generate_resident_injury_events)


@api_router.get("/events/resident-treatment-events", response_model=List[ResidentTreatmentEvent], tags=["events"])
def get_resident_treatment_events(from_date: dt.date, to_date: dt.date):
    validate_dates(from_date, to_date)
    return get_events_date_range(from_date, to_date, get_generated_resident_treatment_events)


@api_router.get("/events/nurse-room-events", response_model=List[NurseRoomEvent], tags=["events"])
def get_nurse_room_events(from_date: dt.date, to_date: dt.date):
    validate_dates(from_date, to_date)
    return get_events_date_range(from_date, to_date, generate_nurse_room_events)


# endregion get events

app = FastAPI()
app.include_router(api_router)
